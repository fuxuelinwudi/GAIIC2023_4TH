# coding:utf-8

import os

import numpy as np
import pandas as pd
import numpy

data_path = './data'

data_path = '../../input/nlp2030'

train_path = os.path.join(data_path, 'train.csv')
test_a_path = os.path.join(data_path, 'preliminary_a_test.csv')

data_path = '../../input/semi7010'
semi_train_path = os.path.join(data_path, 'semi_train.csv')

train_df, test_a_df = pd.read_csv(train_path, header=None), pd.read_csv(test_a_path, header=None)

semi_train_df = pd.read_csv(semi_train_path, header=None)
train_df.columns = ['index', 'desc', 'dialog']
test_a_df.columns = ['index', 'desc']
semi_train_df.columns = ['index', 'desc', 'dialog', 'clinical']

all_word = set()
all_data = []
for i in range(train_df.shape[0]):
    desc, dialog = train_df.desc.iloc[i], train_df.dialog.iloc[i]
    all_data.append(desc.strip())
    all_data.append(dialog.strip())

    for t in desc.split(' '):
        all_word.add(t)

    for t in dialog.split(' '):
        all_word.add(t)

for i in range(semi_train_df.shape[0]):
    desc, dialog, clinical = semi_train_df.desc.iloc[i], semi_train_df.dialog.iloc[i], semi_train_df.clinical.iloc[i]
    if clinical is np.nan:
        clinical = '0'
    all_data.append(desc.strip())
    all_data.append(dialog.strip())
    all_data.append(clinical.strip())
    for t in desc.split(' '):
        all_word.add(t)

    for t in dialog.split(' '):
        all_word.add(t)
    for t in clinical.split(' '):
        all_word.add(t)

for i in range(test_a_df.shape[0]):
    desc = test_a_df.desc.iloc[i]
    all_data.append(desc.strip())

    for t in desc.split(' '):
        all_word.add(t)

output_pretrain_data_path = 'pretrain/vocab/pretrain_data.txt'
output_vocab_path = 'pretrain/vocab/vocab.txt'

with open(output_pretrain_data_path, 'w', encoding='utf-8') as f:
    for line in all_data:
        f.writelines(line + '\n')

with open(output_vocab_path, 'w', encoding='utf-8') as f:
    for word in all_word:
        f.writelines(word + '\n')
