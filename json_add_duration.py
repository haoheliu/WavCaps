import json
import os
from tqdm import tqdm
import torchaudio
from multiprocessing import Pool

def write_json(my_dict, fname):
    with open(fname, 'w') as outfile:
        json.dump(my_dict, outfile, indent=4)

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
    return data

def process_audio(audio_info):
    try:
        wav_path = audio_info["audio"]
        waveform, sr = torchaudio.load(wav_path)
        duration = waveform.shape[-1] / sr
        audio_info["duration"] = duration
    except Exception as e:
        print(e)
    return audio_info

def process_json(json_file):
    data = load_json(json_file)
    processed_data = []
    with Pool() as pool:
        for processed_audio_info in tqdm(pool.imap(process_audio, data["data"]), total=len(data["data"])):
            processed_data.append(processed_audio_info)
    data["data"] = processed_data
    write_json(data, json_file)

if __name__ == '__main__':
    json_path = "/mnt/bn/lqhaoheliu/metadata/processed/general_audio/audiosetcaps/datafiles_abs_path"
    for file in os.listdir(json_path):
        if not file.endswith(".json"):
            continue
        json_file = os.path.join(json_path, file)
        process_json(json_file)
