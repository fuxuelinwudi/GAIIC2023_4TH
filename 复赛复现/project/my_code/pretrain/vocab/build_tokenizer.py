# coding:utf-8


import os
import argparse
from transformers import BartTokenizerFast, BertTokenizer, BartTokenizer

"""
先做个 vocab.txt
"""

pretrain_data_path = 'pretrain/vocab/pretrain_data.txt'

all_words = set()
with open(pretrain_data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        tokens = line.strip().split(' ')
        for t in tokens:
            all_words.add(t)

all_words = sorted(list(all_words))

print('all_words: ', len(all_words))

special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
for i in range(100):
    special_tokens.append(f'[unused{i+1}]')

print('special_tokens num: ', len(special_tokens))

vocab_path = 'pretrain/vocab/vocab.txt'

with open(vocab_path, 'w', encoding='utf-8') as f:
    for word in special_tokens + list(all_words):
        f.writelines(str(word) + '\n')

tokenizer = BertTokenizer.from_pretrained(vocab_path)

vocab_output_path = 'pretrain/new_tokenizer'
if not os.path.exists(vocab_output_path):
    os.mkdir(vocab_output_path)

tokenizer.save_vocabulary(os.path.join(vocab_output_path, 'vocab.txt'))


"""
DEMO
"""
print(tokenizer.cls_token, tokenizer.cls_token_id)
print(tokenizer.sep_token, tokenizer.sep_token_id)
print(tokenizer.mask_token, tokenizer.mask_token_id)
print(tokenizer.pad_token, tokenizer.pad_token_id)
print(tokenizer.unk_token, tokenizer.unk_token_id)

text = '150 50 107 104 113 110 15 13 31 609 20 18 10'
print(len(text.split(' ')), len(tokenizer(text).input_ids))

