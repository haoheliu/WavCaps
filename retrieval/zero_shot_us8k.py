import glob
import json
import torch
import numpy as np

import os
import ipdb
os.environ["HF_HOME"] = "/mnt/bn/arnold-yy-audiodata/pre_load_models"
os.environ["WANDB_API_KEY"] = "ec145d92a5f32070c0b6fa4c1db97e478bbb221e"
from ruamel.yaml import YAML
from tqdm import tqdm

from data_handling.text_transform import text_preprocess
from models.ase_model import ASE
import torchaudio

from torchaudio.transforms import Resample

down_transform = Resample(orig_freq=44100, new_freq=32000)




os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/bn/arnold-yy-audiodata/pre_load_models"
import sys


# config = "settings/H-B-w10000_pretrain.yaml"
# cp_path = '/mnt/bn/lqhaoheliu/project/WavCaps/retrieval/pretrained.pt'

config = "/mnt/bn/lqhaoheliu/project/WavCaps/retrieval/settings/pretrain_asc_full_15ac.yaml"
cp_path = '/mnt/bn/lqhaoheliu/project/WavCaps/retrieval/outputs/05_25_pretrain_asc_full_ac_lr_5e-05_seed_20/models/ac_best_model.pt'

with open(config, "r") as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)

device = "cuda"

model = ASE(config)
model.to(device)


cp = torch.load(cp_path)
model.load_state_dict(cp['model'])
model.eval()
print("Model weights loaded from {}".format(cp_path))


jsonpath = "urbansound_data.json"
class_index_dict_path = 'UrbanSound8K_class_labels_indices.json'
class_index_dict_nospace = 'UrbanSound8K_class_labels_indices_nospace.json'


# Get the class index dict
class_index_dict_space = {v: k for v, k in json.load(open(class_index_dict_path)).items()}

class_index_dict = {v: k for v, k in json.load(open(class_index_dict_nospace)).items()}

# Get all the data
audio_files = []
json_rank = []
data_json = [json.loads(line) for line in open(jsonpath, 'r')]

# data_json = data_json[:800]
for each in data_json:
    audio_files.append(each["wav"])
    json_rank.append(each["label"])
ground_truth_idx = [class_index_dict[jf] for jf in json_rank]


# ipdb.set_trace()



with torch.no_grad():
    ground_truth = torch.tensor(ground_truth_idx).view(-1, 1)

    # Get text features
    all_texts = ["This is a sound of " + t for t in class_index_dict.keys()]

    if len(audio_files)> 400: 
        audio_embed_list = []
        if len(audio_files) % 400 ==0:
            times = len(audio_files)//400 
        else:
            times = len(audio_files)//400 +1
        for time in tqdm(range(times)):
            cur_audio_file = audio_files[time*400:(time+1)*400]
            audio_list = []
            for each_audio in cur_audio_file:
                read_audio = torchaudio.load(each_audio)[0]

                # ipdb.set_trace()
                read_audio = down_transform(read_audio).reshape(-1)
                if read_audio.shape[0]<320000:
                    read_audio = torch.cat([read_audio, torch.zeros(320000-read_audio.shape[0])], dim=0)
                else:
                    read_audio = read_audio[:320000]
                audio_list.append(read_audio.reshape(1,-1))
            all_audio = torch.cat(audio_list, dim=0).reshape(len(cur_audio_file),-1).cuda()
            cur_embed = model.encode_audio(all_audio)
            audio_embed_list.append(cur_embed.cpu().numpy())
            # ipdb.set_trace()
        audio_embed = np.concatenate(audio_embed_list)
    else:
        audio_list = []
        for each_audio in audio_files:
            audio_list.append(torchaudio.load(each_audio)[0].unsqueeze(0))
        all_audio = torch.cat(audio_list, dim=0).reshape(len(audio_files),-1).cuda()
        audio_embed = model.encode_audio(all_audio)
    text_embed = model.encode_text(all_texts).cpu().numpy()

    # text_embed = model.get_text_embedding(all_texts)
    # audio_embed = model.get_audio_embedding_from_filelist(x=audio_files)

    ranking = torch.argsort(torch.tensor(audio_embed) @ torch.tensor(text_embed).t(), descending=True)

    preds = torch.where(ranking == ground_truth)[1]
    preds = preds.cpu().numpy()

    metrics = {}
    metrics[f"mean_rank"] = preds.mean() + 1
    metrics[f"median_rank"] = np.floor(np.median(preds)) + 1
    for k in [1, 5, 10]:
        metrics[f"R@{k}"] = np.mean(preds < k)
    # map@10
    metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

    print(
        f"Zeroshot Classification Results: "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )
    