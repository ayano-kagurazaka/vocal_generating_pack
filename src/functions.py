import binascii
import struct
import base64
import os
import librosa
import soundfile
from Crypto.Cipher import AES
from demucs import separate
from torch.cuda import is_available
from tqdm import tqdm
from pathlib import Path
import json

from classes import DemucsGenerateParam
from environment import output_path, config, so_vits_dataset_path
from so_vits_svc_fork.inference.main import infer
from so_vits_svc_fork.preprocessing.preprocess_flist_config import preprocess_config
from pyannote.audio import Pipeline
from df.enhance import enhance, init_df, load_audio, save_audio
from Slicer import Slicer

def convert_ncm(file_path:Path, output_path:Path) -> Path:
    """
    convert NetEase ncm file to plain sound file
    :param file_path: the path of the ncm file
    :param output_path: the path of the output directory
    :return: the path of the output file
    """
    if os.path.splitext(file_path)[-1] != '.ncm':
        return Path(file_path)
    core_key = binascii.a2b_hex("687A4852416D736F356B496E62617857")
    meta_key = binascii.a2b_hex("2331346C6A6B5F215C5D2630553C2728")
    unpad = lambda s: s[0:-(s[-1] if type(s[-1]) == int else ord(s[-1]))]
    f = open(file_path, "rb")
    header = f.read(8)
    assert binascii.b2a_hex(header) == b'4354454e4644414d'
    f.seek(2,1)
    key_length = f.read(4)
    key_length = struct.unpack("<I", bytes(key_length))[0]
    key_data = f.read(key_length)
    key_data_array = bytearray(key_data)
    for i in range(0, len(key_data_array)):
        key_data_array[i] ^= 0x64
    key_data = bytes(key_data_array)
    cryptor = AES.new(core_key, AES.MODE_ECB)
    key_data = unpad(cryptor.decrypt(key_data))[17:]
    key_length = len(key_data)
    key_data = bytearray(key_data)
    key_box = bytearray(range(256))
    c = 0
    last_byte = 0
    key_offset = 0
    for i in range(256):
        swap = key_box[i]
        c = (swap + last_byte + key_data[key_offset]) & 0xff
        key_offset += 1
        if key_offset >= key_length:
            key_offset = 0
        key_box[i] = key_box[c]
        key_box[c] = swap
        last_byte = c
    meta_length = f.read(4)
    meta_length = struct.unpack("<I", bytes(meta_length))[0]
    meta_data = f.read(meta_length)
    meta_data_array = bytearray(meta_data)
    for i in range(0, len(meta_data_array)):
        meta_data_array[i] ^= 0x63
    meta_data = bytes(meta_data_array)
    meta_data = base64.b64decode(meta_data[22:])
    cryptor = AES.new(meta_key, AES.MODE_ECB)
    meta_data = unpad(cryptor.decrypt(meta_data)).decode("utf-8")[6:]
    meta_data = json.loads(meta_data)
    crc32 = f.read(4)
    crc32 = struct.unpack("<I", bytes(crc32))[0]
    f.seek(5, 1)
    image_size = f.read(4)
    image_size = struct.unpack('<I', bytes(image_size))[0]
    image_data = f.read(image_size)
    file_name = file_path.stem + "." + meta_data["format"]
    Path(output_path).mkdir(parents=True, exist_ok=True)
    m = open(os.path.join(output_path, file_name), "wb+")
    chunk = bytearray()
    while True:
        chunk = bytearray(f.read(0x8000))
        chunk_length = len(chunk)
        if not chunk:
            break
        for i in range(1, chunk_length+1):
            j = i & 0xff
            chunk[i-1] ^= key_box[(key_box[j] + key_box[(key_box[j] + j) & 0xff]) & 0xff]
        m.write(chunk)
    m.close()
    f.close()
    return Path(os.path.join(output_path, file_name))

def separate_vocal(
        track_path: Path,
        output_path: Path,
        save_to_config=False,
        name="",
        device="cpu" if not is_available() else "cuda",
        wav_store_method="float32",
        split_mode="segment",
        split_num=5,
        clip_mode="clamp",
        jobs=os.cpu_count(),
        repo=r"../files/models/demucs/hdemucs_mmi",
        extension="wav"
) -> dict[str, Path]:
    """
    separate the music into vocals and instruments
    :param track_path: the path of the track
    :param output_path: the path of the output directory
    :param device: the device to use, cuda or cpu, default is cpu
    :param wav_store_method: the method to store the wav file, float32 or int16, default is
    :param split_mode: the method to split the track, --segment or --no-split
    :param split_num: the number of segments to split the track, only works when split_mode is --segment
    :param clip_mode: the method to clip the track, rescale or clamp, default is rescale
    :param jobs: the number of jobs to use, default is use all the cpu logic core if in cpu mode
    :param repo: the repo to download the model, default is https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/
    :param save_to_config: whether save the config of this function to a file
    :param name: the name of the config file
    :param extension: extension of output file, default is mp3
    """
    lossy = ["mp3", "m4a", "ogg", "aac"]
    lossless = ["flac", "wav"]

    if extension not in lossy and extension not in lossless:
        raise ValueError("extension must be one of mp3, m4a, ogg, aac, flac, wav")

    split_mode = split_mode if split_mode in ["segment", "no-split"] else "segment"

    args = [str(track_path.resolve()),
         "-o", str(output_path.resolve()),
         "--repo", str(repo),
         "--device", device if device in ["cpu", "cuda"] else "cpu",
         "--" + wav_store_method if wav_store_method in ["float32", "int16"] else "--float32",
         "--" + split_mode, None if split_mode == "no-split" else str(split_num),
         "--clip-mode", clip_mode if clip_mode in ["rescale", "clamp"] else "rescale",
         "--name", "hdemucs_mmi",
         "--jobs", str(0) if jobs < 0 else str(jobs),
         "--two-stems", "vocals",
         "--mp3" if extension in lossy else None
         ]

    if save_to_config:
        import pickle
        args.extend([save_to_config, split_num, name])
        conf = DemucsGenerateParam.from_list(args.copy(), extension)
        conf.get.save_as(name if name != "" else str(int(pickle.dumps(config)) ** 2))
    try:
        args = list(filter((None).__ne__, args))
        args = args[:args.index(save_to_config)]
    except ValueError:
        pass
    print(args)
    separate.main(args)

    if not (extension in lossless and extension != "wav") or not (extension in lossy and extension != "mp3"):
        print("converting file to " + extension + "...")
        if extension in lossless:
            cmd = "ffmpeg -v quiet -threads + " + str(os.cpu_count() // 2) + " -i" + str(Path(os.path.join(output_path, "vocals.wav")).resolve()) + " " + str(Path(os.path.join(output_path, "vocals." + extension.strip("."))).resolve())
        else:
            cmd = "ffmpeg -v quiet -threads " + str(os.cpu_count() // 2) + " -i " + str(Path(os.path.join(output_path, "vocals.mp3")).resolve()) + " " + str(Path(os.path.join(output_path, "vocals." + extension.strip("."))).resolve())
        os.system(cmd)
        print("done")
    output_vocal = Path(output_path).joinpath("hdemucs_mmi").joinpath(track_path.stem).joinpath("vocals." + extension.strip("."))
    output_no_vocal = Path(output_path).joinpath("hdemucs_mmi").joinpath(track_path.stem).joinpath("no_vocals." + extension.strip("."))
    return {"vocal": output_vocal, "instrumental": output_no_vocal}


def separate_vocal_parameterized(param: DemucsGenerateParam) -> dict[str, Path]:
    return separate_vocal(
        track_path=param.data.track_path,
        output_path=param.data.output_path,
        save_to_config=param.data.save_to_config,
        name=param.data.name,
        device=param.data.device,
        wav_store_method=param.data.wav_store_method,
        split_mode=param.data.split_mode,
        split_num=param.data.split_num,
        clip_mode=param.data.clip_mode,
        jobs=param.data.jobs,
        repo=param.data.repo,
        extension=param.data.extension
    )


def apply_so_vits(input_vocal: Path,
                  output_path: Path,
                  model_path: Path,
                  config_file_path: Path,
                  speaker: str,
                  cluster=None,
                  db_threshold=-35,
                  auto_predict_f0=True,
                  noice_scale=0.4,
                  pad_seconds=0.5,
                  f0_method="dio",
                  chunk_seconds=0.5,
                  max_chunk_seconds=40,
                  cluster_infer_ratio=0,
                  absolute_tresh=True,
                  save_to_config=False,
                  name="",
                  ) -> Path:
    """
    :param input_vocal: the path of the extracted vocal
    :param output_path: the path of the output directory
    :param model_path: the path of the model
    :param config_file_path: the path of the model config file
    :param speaker: the speaker of the vocal
    :param cluster: the cluster model for the vocal, optional
    :param db_threshold: the db threshold to remove the noice, default is -35
    :param auto_predict_f0: whether to auto predict the f0, default is True
    :param noice_scale: the scale of the noice, default is 0.4
    :param pad_seconds: the seconds to pad the vocal, default is 0.5
    :param f0_method: the method to predict the f0, default is dio
    :param chunk_seconds: length of each vocal chunk, default is 0.5
    :param max_chunk_seconds: max length of each vocal chunk, default is 40
    :param cluster_infer_ratio: the ratio to infer the cluster, default is 0
    :param save_to_config: whether save the config of this function to a file
    :param name: the name of the config file
    """
    clamp = lambda num, low, high: min(high, max(num, low))
    db_threshold = clamp(db_threshold, 0., -60.)
    noice_scale = clamp(noice_scale, 0., 1.)
    pad_seconds = clamp(pad_seconds, 0., 1.)
    chunk_seconds = clamp(chunk_seconds, 0., 3.)
    max_chunk_seconds = clamp(max_chunk_seconds, 0., 240.)
    cluster_infer_ratio = clamp(cluster_infer_ratio, 0., 1.)
    print("model path: ", model_path)
    f0_method = f0_method if f0_method in ["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] else "dio"
    if not input_vocal.exists():
        raise FileNotFoundError(f"File {input_vocal} not found")
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found")
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config {config_file_path} not found")
    if cluster is not None and not cluster.exists():
        raise FileNotFoundError(f"Cluster model {cluster} not found")

    if speaker not in json.load(open(config_file_path))["spk"]:
        raise ValueError(f"Speaker {speaker} not found in config {config_file_path}")

    output_file = output_path / Path(f"voice_generated_with_{speaker}.wav").name
    infer(
        input_path=input_vocal,
        output_path=output_file,
        model_path=model_path,
        config_path=config_file_path,
        speaker=speaker,
        cluster_model_path=cluster,
        auto_predict_f0=auto_predict_f0,
        db_thresh=db_threshold,
        noise_scale=noice_scale,
        pad_seconds=pad_seconds,
        f0_method=f0_method,
        chunk_seconds=chunk_seconds,
        max_chunk_seconds=max_chunk_seconds,
        cluster_infer_ratio=cluster_infer_ratio,
        absolute_thresh=absolute_tresh
    )
    path_out = output_path / Path(f"voice_generated_with_{speaker}.wav").name
    return path_out


def fuse_vocal_and_instrumental(
        vocal_path: Path,
        instrumental_path: Path,
        output_path: Path,
        speaker: str,
        extension="wav"):
    """
    :param vocal_path: the path of the vocal
    :param instrumental_path: the path of the instrumental
    :param output_path: the path of the output file
    :param speaker: the speaker of the vocal
    :param extension: the extension of the output file, default is wav
    """
    if not vocal_path.exists():
        raise FileNotFoundError(f"File {vocal_path} not found")
    if not instrumental_path.exists():
        raise FileNotFoundError(f"File {instrumental_path} not found")
    output_path.mkdir(parents=True, exist_ok=True)
    
    vocal, sr_vocal = librosa.load(str(vocal_path.resolve()), sr=44100)
    instrumental, sr_instrumental = librosa.load(str(instrumental_path.resolve()), sr=44100)
    audio = instrumental + vocal
    soundfile.write(output_path / Path(vocal_path.stem + f"_counterfeited_from_{speaker}." + extension.strip(".")).name, audio, sr_instrumental, format=extension.strip("."))
    if not Path(output_path / Path(vocal_path.stem + f"_counterfeited_from_{speaker}." + extension.strip(".")).name).exists():
        print(output_path / Path(vocal_path.stem + f"_counterfeited_from_{speaker}." + extension.strip(".")).name)
    print("done")
    return output_path / Path(vocal_path.stem + f"_counterfeited_from_{speaker}." + extension.strip(".")).name


def extract_video_audio(video_path: Path, dir_out: Path, desired_sample_rate=None) -> Path:
    dir_out.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -hide_banner -i {video_path} {dir_out / f'{video_path.stem}_audio.wav '} -y"
    os.system(cmd)
    while not (dir_out / f"{video_path.stem}_audio.wav").exists():
        pass
    return dir_out / f"{video_path.stem}_audio.wav"


def resample(input_path: Path, output_path: Path, sample_rate: int=44100):
    """
    :param input_path: the path of the input file
    :param output_path: the path of the output file
    :param sample_rate: the sample rate of the output file
    :return: the path of the output file
    """
    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} not found")
    output_path.mkdir(parents=True, exist_ok=True)
    audio, sr = librosa.load(input_path)
    librosa.resample(audio, sr, sample_rate)
    soundfile.write(output_path.joinpath(f"{input_path.stem}_resampled_{sample_rate}{input_path.suffix}").resolve(), audio, sample_rate, format='wav')
    return output_path.joinpath(f"{input_path.stem}_resampled_{sample_rate}{input_path.suffix}").resolve()


def denoise(input_path: Path, path_out: Path, sample_rate: int=44100):
    """
    :param input_path: the path of the input file
    :param output_path: the path of the output file
    :param sample_rate: the sample rate of the output file
    :return: the path of the output file
    """
    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} not found")
    output_path.mkdir(parents=True, exist_ok=True)
    model, df_state, _ = init_df()
    audio, _ = load_audio(str(input_path.resolve()), sr=sample_rate)
    denosied_audio = enhance(model, df_state, audio)
    save_audio(str(output_path.joinpath(f"{input_path.stem}_denoised_{sample_rate}{input_path.suffix}").resolve()), denosied_audio, sample_rate)
    return output_path.joinpath(f"{input_path.stem}_denoised_{sample_rate}{input_path.suffix}").resolve()


def extract_speaker(
        input_path: Path,
        path_out: Path,
        sr:int = 44100,
        min_speaker: int = 1,
        max_speaker: int = 1,
        huggingface_token: str = config["keys"]["huggingface_auth"]
) -> Path:
    input_path = Path(input_path)
    path_out = Path(path_out)
    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} not found")
    if not path_out.exists():
        path_out.mkdir(parents=True, exist_ok=True)
    path_out.joinpath(input_path.stem).mkdir(parents=True, exist_ok=True)
    audio = librosa.load(input_path, sr, mono=False)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=huggingface_token)
    diarization = pipeline(input_path, min_speaker=min_speaker, max_speaker=max_speaker)
    speaker_count = {}
    for segment, track, speaker in tqdm(
        list(diarization.itertracks(yield_label=True)), desc="separating"
    ):
        if segment.end - segment.start < 1:
            continue
        speaker_count[speaker] += 1
        audio_cut = audio[int(segment.start * sr): int(segment.end * sr)]
        soundfile.write(
            output_path.joinpath(input_path.stem) / f"{speaker}_{speaker_count[speaker]}.wav",
            audio_cut,
            sr
        )

    return path_out.joinpath(input_path.stem)


def slice_audio(
        input_path: Path,
        path_out: Path = None,
        db_threshold: float = -40,
        min_len_ms: int = 1000,
        min_silence_interval_ms: int = 300,
        hop_length_ms: int = 10,
        max_silence_len_ms: int = 500,
        extension: str = "wav",
        desired_samplerate:int = 44100
) -> Path:
    """
    :param input_path: the path of the input file
    :param path_out: the path of the output directory, default is [the path you defined in the config]/sliced
    :param db_threshold: the db threshold for silence detection in ms, default is -40
    :param min_len_ms: the min length of each slice in ms, default is 1000
    :param min_silence_interval_ms: the min silence interval in ms, default is 300
    :param hop_length_ms: the frame in ms, default is 10
    :param max_silence_len_ms: the max silence length in ms, default is 500
    :param extension: the extension of the output file, default is wav
    :param desired_samplerate: the desired sample rate of the output file, default is 44100
    """
    if path_out is None:
        path_out = so_vits_dataset_path.joinpath(input_path.stem).joinpath("sliced")
    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} not found")
    if librosa.get_samplerate(input_path) != desired_samplerate:
        t = resample(input_path, Path(input_path.stem + f"_resampled_{desired_samplerate}" + input_path.suffix), desired_samplerate)
        os.remove(input_path)
        input_path = t
    path_out.mkdir(parents=True, exist_ok=True)
    audio, sr = librosa.load(input_path, sr=None, mono=False)
    slicer = Slicer(
        sr=sr,
        threshold=db_threshold,
        min_length=min_len_ms,
        min_interval=min_silence_interval_ms,
        hop_size=hop_length_ms,
        max_sil_kept=max_silence_len_ms
    )
    chunks = slicer.slice(audio)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T
        soundfile.write(path_out.joinpath(input_path.stem + f"_{i}th_slice" + f".{extension}"), chunk, sr)
    return path_out

def generate_config(
        sliced_path: Path,
        train_data_path: Path = None,
        val_data_path: Path = None,
        test_data_path: Path = None,
        config_file_path: Path = None,
        config_name: str = "config.json"
):
    """
    :param sliced_path: the path to the separated dataset folder
    :param train_data_path: the output path of training dataset
    :param val_data_path: the output path of validation dataset
    :param test_data_path: the output path of testing dataset
    :param config_file_path: the path of the output config file
    :param config_name: the name of the output config file
    """
    if train_data_path is None:
        train_data_path = so_vits_dataset_path.joinpath(sliced_path.stem).joinpath("train")
        train_data_path.mkdir(parents=True, exist_ok=True)
    if val_data_path is None:
        val_data_path = so_vits_dataset_path.joinpath(sliced_path.stem).joinpath("val")
        val_data_path.mkdir(parents=True, exist_ok=True)
    if test_data_path is None:
        test_data_path = so_vits_dataset_path.joinpath(sliced_path.stem).joinpath("test")
        test_data_path.mkdir(parents=True, exist_ok=True)
    if config_file_path is None:
        config_file_path = so_vits_dataset_path.joinpath(sliced_path.stem).joinpath(config_name)
        so_vits_dataset_path.joinpath(sliced_path.stem).joinpath(config_name).touch(exist_ok=True)

    preprocess_config(sliced_path, train_data_path, val_data_path, test_data_path, config_file_path, config_name)

    return {"train": train_data_path, "val": val_data_path, "test": test_data_path, "config": config_file_path}

