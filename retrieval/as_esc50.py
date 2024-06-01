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


os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/bn/arnold-yy-audiodata/pre_load_models"
import sys


# config = "/mnt/bn/lqhaoheliu/project/WavCaps/retrieval/settings/pretrain_asc_full_batchsize_384.yaml"
config = "settings/H-B-w10000_pretrain.yaml"

# cp_path = '/mnt/bn/lqhaoheliu/project/WavCaps/retrieval/outputs/05_21_pretrain_asc_full_batch_384_lr_5e-05_seed_20/models/best_model.pt'
cp_path = "/mnt/bn/lqhaoheliu/project/WavCaps/retrieval/pretrained.pt"

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




esc50_test_dir = '/mnt/bn/arnold-yy-audiodata/audio_data/esc50/clap_data/test/'
class_index_dict_path = '/mnt/bn/arnold-yy-audiodata/clap_new/CLAP-main/class_labels/ESC50_class_labels_indices_space.json'



# Get the class index dict
class_index_dict = {v: k for v, k in json.load(open(class_index_dict_path)).items()}

# Get all the data
audio_files = sorted(glob.glob(esc50_test_dir + '**/*.flac', recursive=True))
json_files = sorted(glob.glob(esc50_test_dir + '**/*.json', recursive=True))
ground_truth_idx = [class_index_dict[json.load(open(jf))['tag'][0]] for jf in json_files]

with torch.no_grad():
    ground_truth = torch.tensor(ground_truth_idx).view(-1, 1).cuda()

    # Get text features
    all_texts = ["This is a sound of " + t for t in class_index_dict.keys()]

    audio_list = []
    for each_audio in audio_files:
        audio_list.append(torchaudio.load(each_audio)[0].unsqueeze(0))
    all_audio = torch.cat(audio_list, dim=0).reshape(len(audio_files),-1).cuda()
    # ipdb.set_trace()
    audio_embed = model.encode_audio(all_audio)
    text_embed = model.encode_text(all_texts)

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
    