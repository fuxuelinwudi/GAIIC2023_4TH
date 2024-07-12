# coding:utf-8
import argparse
import os
import torch
import random
import numpy as np
from transformers import BertTokenizer, BartTokenizer
import sys
sys.path.append('.')
from modeling.modeling_bart import BartForConditionalGeneration
from modeling.modeling_pagasus import PegasusForConditionalGeneration
from modeling.modeling_cpt import CPTForConditionalGeneration


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


seed_everything(2023)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='./pretrain/pretrain_model/bart-base-chinese', type=str, help='')
parser.add_argument('--tokenizer_path', default='./pretrain/new_tokenizer', type=str, help='')
parser.add_argument('--save_path', default='./pretrain/pretrain_model/new-bart-base', type=str, help='')
parser.add_argument('--model_name', default='bart', type=str, help='')

args = parser.parse_args()
model_dic = {
    'bart': BartForConditionalGeneration,
    'cpt': CPTForConditionalGeneration,
    'pegasus': PegasusForConditionalGeneration,
}
print('>>> now model name: ', args.model_name)
model_path = args.model_path
old_tokenizer = BertTokenizer.from_pretrained(model_path)
new_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

model = model_dic[args.model_name].from_pretrained(model_path)
model.resize_token_embeddings(old_tokenizer, new_tokenizer)

model.config.bos_token_id = new_tokenizer.cls_token_id
model.config.eos_token_id = new_tokenizer.sep_token_id
model.config.forced_bos_token_id = new_tokenizer.cls_token_id
model.config.forced_eos_token_id = new_tokenizer.sep_token_id
model.config.pad_token_id = new_tokenizer.pad_token_id
model.config.decoder_start_token_id = new_tokenizer.sep_token_id

save_path = args.save_path
model.save_pretrained(save_path)
new_tokenizer.save_vocabulary(os.path.join(save_path, 'vocab.txt'))
# print(model)
