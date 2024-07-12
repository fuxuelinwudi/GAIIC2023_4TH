# coding:utf-8
import argparse
import os
import copy
import torch
import random
import numpy as np
from transformers import BertTokenizer
from configurations.train_config import Config
import sys
sys.path.append('.')
from modeling.modeling_cpt import CPTForConditionalGeneration
from modeling.modeling_pagasus import PegasusForConditionalGeneration
from modeling.modeling_bart import BartForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2023, type=int, help='')
parser.add_argument('--model_name', default='bart', type=str, help='')
parser.add_argument('--save_path', default='./finetune/finetune_models/bart_finetuned_model-base-dae', type=str, help='')
parser.add_argument('--pretrain_model_path',
                    default='./pretrain/pretrain_model/post-pretrained-bart-base-dae/epoch_120',
                    type=str, help='')
parser.add_argument('--swa_output_dir',
                    default='./finetune/swa_finetune_models/bart_finetuned_model_swa_base_ade_output',
                    type=str, help='')

args = parser.parse_args()

user_args = Config()
user_args.pretrain_model_path = args.pretrain_model_path
user_args.seed = args.seed
user_args.save_path = args.save_path
user_args.swa_output_dir = args.swa_output_dir
if not os.path.exists(user_args.swa_output_dir):
    os.mkdir(user_args.swa_output_dir)
model_dic = {
    'bart': BartForConditionalGeneration,
    'cpt': CPTForConditionalGeneration,
    'pegasus': PegasusForConditionalGeneration,
}

def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def get_model_path_list(base_dir):
    model_lists = []

    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if 'pytorch_model.bin' in _file:
                model_lists.append(os.path.join(root, _file).replace("\\", '/'))
    print(model_lists)

    model_lists = sorted(model_lists,
                         key=lambda x: int(x.split('/')[-2].split('_')[1]))
    return model_lists


def swa(model, tokenizer, model_dir, swa_output_dir, swa_start=1, swa_end=0):
    model_path_list = get_model_path_list(model_dir)

    # assert 1 <= swa_start < len(model_path_list) - 1, \
    #     f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    if swa_end == 0:
        select_model_path_list = model_path_list[swa_start - 1:]
    else:
        select_model_path_list = model_path_list[swa_start - 1: swa_end]

    print(select_model_path_list)
    with torch.no_grad():
        for _ckpt in select_model_path_list:
            print(_ckpt)
            state_dict = torch.load(_ckpt, map_location=user_args.device)
            model.load_state_dict(state_dict)
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = os.path.join(swa_output_dir)
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    swa_model_path = os.path.join(swa_model_dir)

    swa_model.save_pretrained(swa_model_path)
    tokenizer.save_pretrained(swa_model_path)

    return swa_model


def build_model():
    model = model_dic[args.model_name].from_pretrained(user_args.pretrain_model_path)
    model.to(user_args.device)
    tokenizer = BertTokenizer.from_pretrained(user_args.pretrain_model_path)
    return model, tokenizer


def main():

    seed_everything(user_args.seed)

    swa_raw_model, tokenizer = build_model()

    swa(swa_raw_model, tokenizer,
        user_args.save_path,
        swa_output_dir=user_args.swa_output_dir,
        swa_start=user_args.swa_start,
        swa_end=user_args.swa_end)


if __name__ == '__main__':
    main()


