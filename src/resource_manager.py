import tqdm
from environment import demucs_model_path, so_vits_model_path, config
from utilities import update_download_path, get_cow_transfer_file, get_hugging_face_file
from environment import sources
from classes import AttributeDict
from pathlib import Path
import requests
import json



def get_data_from_source(engine_name: str, file_type: str, file_name: str, update_cache=False):
	"""
	get model from source
	"""
	download_result = AttributeDict()
	try:
		model_download_data_dict = sources.get_attribute(f"{engine_name}.{file_type}.{file_name}", sources, strict=True)
	except KeyError as e:
		raise e

	match engine_name:
		case "demucs":
			for i in model_download_data_dict["link"]:
				download_result.update(get_demucs_model(model_name=file_name, link=i, download_path=demucs_model_path, update_cache=update_cache, auth=model_download_data_dict["auth"]))
		case "so-vits":
			for i in model_download_data_dict["link"]:
				download_result.update(get_so_vits_model(model_name=file_name, link=i, download_path=so_vits_model_path, update_cache=update_cache, auth=model_download_data_dict["auth"]))
		case _:
			print(f"engine {engine_name} not supported, skipping")

	return download_result


def list_available_resources(engine_name, file_type):
	"""
	list all available resources for a certain file_type of a engine from sources.json
	"""
	try:
		return list(sources.get_attribute(f"{engine_name}.{file_type}", strict=True).keys())
	except KeyError as e:
		raise e


def get_demucs_model(model_name:str, link:str, download_path:Path, update_cache:bool, auth:dict) -> dict:
	"""
	get demucs demo_assets from config.json
	"""
	download_path.joinpath(model_name).mkdir(parents=True, exist_ok=True)
	print(download_path.joinpath(model_name))
	for j in download_path.joinpath(model_name).iterdir():
		if link.split('/')[-1] in list(Path(demucs_model_path).joinpath(model_name).iterdir()) and not update_cache:
			print(f"{link.split('/')[-1]} already exists, skipping")
			return {link.split('/')[-1]: j.joinpath(link.split('/')[-1]).resolve()}
	content = requests.get(link, stream=True)
	length = int(content.headers["content-length"])
	print(f"Downloading {link.split('/')[-1]} ({length} bytes)")
	with tqdm.tqdm_notebook(desc=link.split('/')[-1], total=length, unit="iB", unit_scale=True) as pbar:
		with open((download_path.joinpath(model_name).joinpath(link.split('/')[-1])), "wb") as f:
			for chunk in content.iter_content(chunk_size=1024):
				if chunk:
					f.write(chunk)
					pbar.update(len(chunk))
	print(f"{link.split('/')[-1]} downloaded")
	update_download_path("demucs", "model", model_name, link.split('/')[-1], download_path.joinpath(model_name).joinpath(link.split('/')[-1]))
	return {link.split('/')[-1]: download_path.joinpath(model_name).resolve()}


def get_so_vits_model(model_name, download_path:Path, link:str, auth:dict, update_cache=True) -> dict:
	"""
	get so-vits-svc demo_assets from sources.json
	"""

	download_path.joinpath(model_name).mkdir(parents=True, exist_ok=True)
	if link.startswith("https://cowtransfer.com/"):
		return get_cow_transfer_file(model_name, link, download_path,"model","so-vits")
	elif link.startswith("https://huggingface.co/"):
		return get_hugging_face_file(model_name, link, download_path, "so-vits", "model", ["*.pt", "*.pth", "*.json"], update_cache, auth)
	else:
		print(f"unknown model source {link}, currently only support cow transfer and huggingface")
		return {}



def export_sources(ignore_private:bool=False):
	"""
	export sources.json to sources_export.json, remove private sources and local paths
	"""
	def traverse_dict_remove_private(dict_in:dict):
		ret = {}
		for i in dict_in:
			if isinstance(dict_in[i], dict):
				if i == "local":
					ret[i] = {}
				elif not (dict_in[i].get("private", False) and dict_in[i].get("private", False)) or ignore_private:
					ret[i] = traverse_dict_remove_private(dict_in[i])
			else:
				ret[i] = dict_in[i]
		return ret
	ans = traverse_dict_remove_private(sources.__dict__)
	json.dump(ans, open(config["sources_export"], "w+"), indent=4)


def get_all_models(update_cache=False):
	"""
	download all models from sources.json
	"""
	ans = {}
	for i in sources:
		for j in sources[i]["models"]:
			ans.update(get_data_from_source(i, "model", j, update_cache=update_cache))
	return ans


def get_all_datasests(update_cache=False):
	raise NotImplementedError


def get_all_resources(update_cache=False):
	"""
	get all resources from sources.json
	"""
	get_all_models(update_cache=update_cache)

