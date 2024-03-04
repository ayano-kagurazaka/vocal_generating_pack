import os
from functions import separate_vocal, apply_so_vits, fuse_vocal_and_instrumental, convert_ncm, Path
from resource_manager import get_data_from_source
from environment import config
model_path = Path("demo_assets/models/").resolve()

model_path = get_data_from_source("so-vits", "model", "nahida-jp", update_cache=True)

path = Path(config["output"]) / Path("demo")
converted_path = convert_ncm("demo_assets/demo.wav", Path(path.stem)/Path("./converted/"))
print("converted")
separated_path = separate_vocal(converted_path, Path(path.stem)/Path("./separated/"), jobs=os.cpu_count() - 1)
print("separated")
models = get_avaliable_so_vits_models_dict()
inferred_path = apply_so_vits(input_vocal=Path(separated_path["vocal"]),
							  output_path=Path(path.stem) / Path("./inferred/"), model_path=Path(
		models["nahida-jp"].joinpath("nahida_jp_G_40000.pth")
	), config_file_path=Path(
		models["nahida-jp"].joinpath("nahida.json")
	), speaker="nahida", cluster=Path(
		models["nahida-jp"].joinpath("nahida_jp_kmeans_10000.pt")
	), auto_predict_f0=False)
print("inferred")
print(fuse_vocal_and_instrumental(
	vocal_path=inferred_path,
	instrumental_path=separated_path["instrumental"],
	output_path=Path(path.stem)/Path("./counterfeited"),
	speaker="nahida"
))
