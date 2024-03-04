from pathlib import Path
import json
from requests import get
import classes

config_path = Path("environment.json").resolve()
if not config_path.exists():
	print("environment.json not found, creating...")
	config_path.touch()
	config_path.resolve().write_text('''
	{
		"model": {
			"demucs": "../resources/files/models/demucs",
			"so-vits": "../resources/files/models/so-vits"
		},
		"preset": {
			"demucs": "../resources/files/presets/demucs",
			"so-vits": "../resources/files/presets/so-vits"
		},
		"dataset": {
			"demucs": "../resources/files/datasets/demucs",
			"so-vits": "../resources/files/datasets/so-vits"
		},
		"output": "../resources/files/output",
		"sources": "../resources/files/sources.json",
		"sources_export": "../resources/files/sources_export.json",
		"key_path": "../resources/files/keys",
		"keys": {}
	}
	''')

# env variables
config = json.load(open(config_path, "r"))
demucs_model_path = Path(config["model"]["demucs"]).resolve()
so_vits_model_path = Path(config["model"]["so-vits"]).resolve()
demucs_preset_path = Path(config["preset"]["demucs"]).resolve()
demucs_dataset_path = Path(config["dataset"]["demucs"]).resolve()
so_vits_preset_path = Path(config["preset"]["so-vits"]).resolve()
so_vits_dataset_path = Path(config["dataset"]["so-vits"]).resolve()
output_path = Path(config["output"]).resolve()
key_path = Path(config["key_path"]).resolve()



# model path
print("building file structures...")
demucs_model_path.mkdir(parents=True, exist_ok=True)
so_vits_model_path.mkdir(parents=True, exist_ok=True)
demucs_preset_path.mkdir(parents=True, exist_ok=True)
so_vits_preset_path.mkdir(parents=True, exist_ok=True)
so_vits_dataset_path.mkdir(parents=True, exist_ok=True)
demucs_dataset_path.mkdir(parents=True, exist_ok=True)
output_path.mkdir(parents=True, exist_ok=True)
key_path.mkdir(parents=True, exist_ok=True)
sources_path = Path(config["sources"]).resolve()
Path("../files").mkdir(exist_ok=True, parents=True)

if not sources_path.exists():
	with open(sources_path, "wb+") as s:
		print("downloading sources.json...")
		s.write(
			get("https://raw.githubusercontent.com/ayano-kagurazaka/vocal_generating_pack/main/files/sources_export.json").content)
try:
	sources = classes.AttributeDict(json.load(open(sources_path, "r")))
except json.decoder.JSONDecodeError:
	print("sources.json is broken, downloading again...")
	with open(sources_path, "wb+") as s:
		s.write(
			get("https://raw.githubusercontent.com/ayano-kagurazaka/vocal_generating_pack/main/files/sources_export.json").content)
	sources = classes.AttributeDict(json.load(open(sources_path, "r")))
print("done")


def update_env():
	global config, sources_path, sources, demucs_model_path, so_vits_model_path, demucs_preset_path, so_vits_preset_path, so_vits_dataset_path, demucs_dataset_path, output_path
	config = json.load(open(config_path, "r"))
	sources_path = Path(config["sources"]).resolve()
	if not sources_path.exists():
		with open(sources_path, "wb+") as s:
			s.write(
				get("https://raw.githubusercontent.com/ayano-kagurazaka/vocal_generating_pack/main/files/sources_export.json").content)
	sources = classes.AttributeDict(json.load(open(sources_path, "r")))
	demucs_model_path = Path(config["model"]["demucs"]).resolve()
	so_vits_model_path = Path(config["model"]["so-vits"]).resolve()
	demucs_preset_path = Path(config["preset"]["demucs"]).resolve()
	so_vits_preset_path = Path(config["preset"]["so-vits"]).resolve()
	so_vits_dataset_path = Path(config["dataset"]["so-vits"]).resolve()
	demucs_dataset_path = Path(config["dataset"]["demucs"]).resolve()
	output_path = Path(config["output"]).resolve()


def update_keys():
	for i in key_path.iterdir():
		with open(i, "r") as k:
			if k.read().strip("\n").strip(" ") != "":
				config["keys"][i.stem] = k.read().strip("\n")
			else:
				print(f"key file {i} is empty")


update_keys()