import os

from requests import get
from pathlib import Path
from zipfile import ZipFile
import tarfile
from py7zr import SevenZipFile
from rarfile import RarFile
from gzip import GzipFile
from filetype import guess_extension
from environment import sources, sources_path, update_env
import json
import shutil
import re
import requests
import huggingface_hub
from typing import Any


def cow_transfer_metadata(link: str) -> dict:
	return get("https://api.kit9.cn/api/nainiu_netdisc/api.php?link=" + link).json()


def extract_zip(path: Path):
	with ZipFile(path, "r") as z:
		z.extractall(path.parent)


def extract_tar(path: Path):
	with tarfile.open(path, "r") as t:
		t.extractall(path.parent)


def extract_rar(path: Path):
	with RarFile(path, "r") as r:
		r.extractall(path.parent)


def extract_7z(path: Path):
	with SevenZipFile(path, "r") as s:
		s.extractall(path.parent)


def extract_gz(path: Path) -> Path:
	file_name = path.stem
	with open(path, "rb") as f:
		with GzipFile(fileobj=f) as g:
			with open(str(path.parent) + file_name + guess_extension(str(path)).lower(), "wb+") as f:
				f.write(g.read())
	return Path(file_name + guess_extension(str(path))).resolve()


def move_file(directory: Path, root_dir: Path):
	for i in directory.iterdir():
		if i.is_dir():
			move_file(i, root_dir)
		elif i.name.rsplit('.')[-1] == "pth":
			for i in directory.iterdir():
				shutil.copy(i, root_dir)


def update_download_path(engine_name: str, file_type: str, model_name: str, file_name: str, download_path: Path):
	if not isinstance(sources[engine_name][file_type][model_name]["local"], dict):
		sources[f"{engine_name}.{file_type}.{model_name}.local"] = {}
	sources.__dict__[engine_name][file_type][model_name]["local"].update({file_name: str(download_path)})
	json.dump(sources.__dict__, open(str(sources_path), "w+"), indent=4)
	update_env()


def update_download_path_dict(engine_name: str, file_type: str, model_name: str, data: dict):
	if not isinstance(sources[engine_name][file_type][model_name]["local"], dict):
		sources[f"{engine_name}.{file_type}.{model_name}.local"] = {}
	sources[engine_name][file_type][model_name]["local"].update(data)
	sources.make_json_able()
	json.dump(sources, open(str(sources_path), "w+"), indent=4)
	update_env()
	return sources_path


def flush_sources_cache(remove_file=True, current_layer: dict = sources):
	for i in current_layer.values():
		if "local" in i and isinstance(i["local"], list):
			while len(i["local"]) > 0:
				removed = i["local"].pop()
				if remove_file:
					try:
						shutil.rmtree(removed)
					except FileNotFoundError:
						print(f"File {removed} not found, it could be moved, renamed or deleted manually")
						continue
				print(f"Removed {removed}")
		elif isinstance(i, dict):
			flush_sources_cache(remove_file, i)
	json.dump(sources, open(sources_path, "w+"), indent=4)


def get_cow_transfer_file(name, value, download_path, engine, type, auth:dict={}):
	download_path.joinpath(name).mkdir(parents=True, exist_ok=True)
	archive_supported = ("gz", "gzip", "zip", "tar", "rar", "7z")
	meta = cow_transfer_metadata(value["link"])
	if meta["code"] != 200:
		print(f"cannot fetch metadata of {name}, skipping")
		return {}
	print(f"Downloading {name} ({meta['data']['file_size']})")
	content = requests.get(meta["data"]["download_link"], stream=True)
	local_path = Path(
		download_path.joinpath(name).joinpath(meta["data"]["file_name"] + "." + meta["data"]["file_format"]))
	with open(local_path, "wb") as f:
		f.write(content.content)
	print(f"Downloaded {name}")
	print(f"Extracting {name}")
	filetype = guess_extension(str(local_path))
	match filetype:
		case "gz", "gzip":
			extracted_path = extract_gz(local_path)
			if extracted_path.suffix == ".tar":
				extract_tar(extracted_path)
		case "zip":
			extract_zip(local_path)
		case "tar":
			extract_tar(local_path)
		case "rar":
			extract_rar(local_path)
		case "7z":
			extract_7z(local_path)
		case _:
			print(f"cannot extract {name}, supported file format: {archive_supported}, skipping")
			return {}
	print(f"Extracted {name}")
	os.remove(local_path)
	update_download_path_dict(engine, type, name, dict(
		zip([i.name for i in download_path.joinpath(name).iterdir()],
			[j.resolve() for j in download_path.joinpath(name).iterdir()])))
	return dict(zip([i.name for i in download_path.joinpath(name).iterdir()],
					[j.resolve() for j in download_path.joinpath(name).iterdir()]))


def get_hugging_face_file(name, value, download_path, engine:str, type:str, patterns:list, update_cache=True, auth:dict={}) -> dict[Any, Any]:
	download_path.mkdir(parents=True, exist_ok=True)
	pattern = r"https:\/\/huggingface\.co\/([-\w.]+)\/([\w.-]+)\/?"
	match = re.match(pattern, value)
	repo_id = match.group(1) + "/" + match.group(2).strip("/")
	print("downloading: ", repo_id)
	huggingface_hub.snapshot_download(repo_id, allow_patterns=patterns, local_dir=download_path.joinpath(name), local_dir_use_symlinks=False,
															  force_download=update_cache, token=auth.get("huggingface", None))
	print("downloaded: ", repo_id)
	key, value = [], []
	for i in download_path.joinpath(name).iterdir():
		for j in i.iterdir():
			key.append(i.name + "/" + j.name)
			value.append(j.resolve())
			update_download_path(engine, type, name, i.name, j.resolve())
	return dict(zip(key, value))



def export_sources():
	raise NotImplementedError


def remove_model(model_name, from_source=False):
	raise NotImplementedError


def add_source(url, model_name=None, download=False):
	raise NotImplementedError


def download_from_source(source_name):
	raise NotImplementedError
