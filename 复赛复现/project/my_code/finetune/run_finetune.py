# coding:utf-8
import argparse
import os
import torch
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from collections import defaultdict
from pycocoevalcap.bleu.bleu import Bleu
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AdamW, get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, BertTokenizerFast
import sys
sys.path.append('.')
from baseline.utils import Smoother
from baseline.evaluate import CiderD
from modeling.modeling_cpt import CPTForConditionalGeneration
from modeling.modeling_pagasus import PegasusForConditionalGeneration
from modeling.modeling_bart import BartForConditionalGeneration
from tricks.ema import EMA
from tricks.Lookahead import Lookahead
from tricks.NovoGrad import NovoGrad
from tricks.rdrop import compute_kl_loss
from tricks.LabelSmoothLoss import label_smoothing_loss
from tricks.adv_for_bart import FGM, PGD


from configurations.train_config import Config

warnings.filterwarnings('ignore')

user_args = Config()

# device = 'cpu'
device = user_args.device

"""微调 """
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=2023, type=int, help='')
parser.add_argument('--smoothing', default=0.12, type=float, help='')
parser.add_argument('--model_name', default='bart', type=str, help='')
parser.add_argument('--save_path', default='./finetune/finetune_models/bart_finetuned_model-base-dae', type=str, help='')
parser.add_argument('--pretrain_model_path',
                    default='./pretrain/pretrain_model/post-pretrained-bart-base-dae/epoch_120',
                    type=str, help='')

args = parser.parse_args()

user_args.pretrain_model_path = args.pretrain_model_path
user_args.seed = args.seed
user_args.save_path = args.save_path
if not os.path.exists(user_args.save_path):
    os.mkdir(user_args.save_path)
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


def read_data(tokenizer) -> dict:
    train_df = pd.read_csv(user_args.train_path, header=None)
    train_df.columns = ['index', 'desc', 'dialog']
    semi_train_df = pd.read_csv(user_args.semi_train_path, header=None)
    semi_train_df.columns = ['index', 'desc', 'dialog', 'clinical']

    inputs = defaultdict(list)

    for i, row in tqdm(train_df.iterrows(), desc='Reading from train data', total=len(train_df)):
        desc, dialog = train_df.desc[i].strip(), train_df.dialog[i].strip()
        inputs['desc'].append(desc)
        inputs['clinical'].append(tokenizer.unk_token)
        inputs['dialog'].append(dialog)

    for i, row in tqdm(semi_train_df.iterrows(), desc='Reading from semi train data', total=len(semi_train_df)):
        desc, dialog = semi_train_df.desc[i].strip(), semi_train_df.dialog[i].strip()
        clinical = semi_train_df.clinical[i]
        inputs['desc'].append(desc)
        if str(clinical) == 'nan':
            inputs['clinical'].append(tokenizer.unk_token)
        else:
            inputs['clinical'].append(clinical)
        inputs['dialog'].append(dialog)

    return inputs


class GAIICDataset(Dataset):
    def __init__(self, data_dict):
        super(Dataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index):
        data = (
            self.data_dict['desc'][index],
            self.data_dict['clinical'][index],
            self.data_dict['dialog'][index],
        )
        return data

    def __len__(self) -> int:
        return len(self.data_dict['desc'])


class GAIICDataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def __call__(self, examples: list):
        batch_desc, batch_clinical, batch_dialog = list(zip(*examples))
        batch_max_length = max([len(desc_item.split(' ')) + len(clinical_item.split(' '))
                                for desc_item, clinical_item in zip(batch_desc, batch_clinical)])
        batch_input_text = [(desc_item, clinical_item) for desc_item, clinical_item in
                            zip(batch_desc, batch_clinical)]
        batch_encode_outputs = self.tokenizer.batch_encode_plus(batch_input_text,
                                                                add_special_tokens=True,
                                                                max_length=batch_max_length + 3,
                                                                padding='max_length',
                                                                truncation=True,
                                                                return_tensors='pt')
        batch_labels_max_length = max([len(item.split(' ')) for item in batch_dialog])

        batch_labels_encode_outputs = self.tokenizer.batch_encode_plus(batch_dialog,
                                                                       add_special_tokens=True,
                                                                       max_length=batch_labels_max_length + 2,
                                                                       padding='max_length',
                                                                       truncation=True,
                                                                       return_tensors='pt')
        batch_labels_input_ids = batch_labels_encode_outputs.input_ids

        # labels : a, b, c, d, e, sep -> decoder input : sep, a, b, c, d, e
        decoder_start_ids = torch.tensor([self.pad_token_id], dtype=torch.long)
        batch_decoder_start_ids = decoder_start_ids.unsqueeze(0).repeat_interleave(len(batch_dialog), dim=0)
        batch_decoder_input_ids = torch.cat([batch_decoder_start_ids, batch_labels_input_ids[:, :-1]], dim=-1)

        # replace label's end token id to pad token id
        indexes = batch_decoder_input_ids == self.sep_token_id
        batch_decoder_input_ids[indexes] = self.pad_token_id

        # replace first token id to sep token id
        batch_decoder_input_ids[:, 0] = self.sep_token_id

        # build labels input ids, replace pad token id to -100
        indexes = batch_labels_input_ids == self.pad_token_id
        batch_labels_input_ids[indexes] = -100

        data_dict = {
            'input_ids': batch_encode_outputs['input_ids'],
            'attention_mask': batch_encode_outputs['attention_mask'].to(torch.float),
            'decoder_input_ids': batch_decoder_input_ids,
            'decoder_attention_mask': torch.ne(batch_decoder_input_ids, 0).to(torch.float),
            'labels': batch_labels_input_ids

        }
        return data_dict


def build_model():
    print(f'>>> Loading model from : {user_args.pretrain_model_path}  !!!')
    model = model_dic[args.model_name].from_pretrained(
        user_args.pretrain_model_path
    )
    model.config.max_length = user_args.encoder_max_length
    model.to(device)
    return model


def build_optimizer(model,
                    total_steps, warmup_steps,
                    lka=False,
                    layer_wise_lr=False, up=True,
                    scheduler_type='linear',
                    mi_estimator=None):
    num_encoder_layers = model.config.encoder_layers
    num_decoder_layers = model.config.decoder_layers

    encoder_lr = user_args.encoder_lr
    decoder_lr = user_args.decoder_lr

    other_lr = user_args.other_lr
    opt_lr = user_args.opt_lr

    eps = user_args.eps
    weight_decay = user_args.weight_decay

    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(model.named_parameters())

    encoder_param_optimizer = []
    decoder_param_optimizer = []
    other_param_optimizer = []

    if user_args.png:
        pointer_lr = user_args.pointer_lr
        pointer_param_optimizer = []
        for name, param in model_param:
            if 'encoder' in str(name):
                encoder_param_optimizer.append((name, param))
            elif 'decoder' in str(name):
                decoder_param_optimizer.append((name, param))
            elif 'generator' in str(name):
                pointer_param_optimizer.append((name, param))
            else:
                other_param_optimizer.append((name, param))
    else:
        for name, param in model_param:
            if 'encoder' in str(name):
                encoder_param_optimizer.append((name, param))
            elif 'decoder' in str(name):
                decoder_param_optimizer.append((name, param))
            else:
                other_param_optimizer.append((name, param))

    # debug
    # for name, param in model_param:
    #     print(name)
    # das

    if layer_wise_lr:
        decay_ratio = 0.95

        encoder_lr_list = []
        for layer in range(num_encoder_layers - 1, -1, -1):
            encoder_lr_list.append(encoder_lr)
            encoder_lr *= decay_ratio

        if up:
            encoder_lr_list.reverse()

        optimizer_grouped_parameters = []
        for layer in range(num_encoder_layers - 1, -1, -1):
            encoder_params = {
                'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n and
                           not any(nd in n for nd in no_decay)],
                'lr': encoder_lr_list[layer],
                "weight_decay": weight_decay
            }
            optimizer_grouped_parameters.append(encoder_params)
            encoder_not_decay_params = {
                'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n and
                           any(nd in n for nd in no_decay)],
                'lr': encoder_lr_list[layer],
                "weight_decay": 0
            }
            optimizer_grouped_parameters.append(encoder_not_decay_params)

        decoder_lr_list = []
        for layer in range(num_decoder_layers - 1, -1, -1):
            decoder_lr_list.append(decoder_lr)
            decoder_lr *= decay_ratio

        if up:
            decoder_lr_list.reverse()

        for layer in range(num_decoder_layers - 1, -1, -1):
            decoder_params = {
                'params': [p for n, p in model.named_parameters() if f'decoder.layers.{layer}.' in n and
                           not any(nd in n for nd in no_decay)],
                'lr': decoder_lr_list[layer],
                "weight_decay": weight_decay
            }
            optimizer_grouped_parameters.append(decoder_params)
            decoder_not_decay_params = {
                'params': [p for n, p in model.named_parameters() if f'decoder.layers.{layer}.' in n and
                           any(nd in n for nd in no_decay)],
                'lr': decoder_lr_list[layer],
                "weight_decay": 0
            }
            optimizer_grouped_parameters.append(decoder_not_decay_params)

        other_params = [
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay, 'lr': other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': other_lr},
        ]
        optimizer_grouped_parameters.extend(other_params)

    else:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay, 'lr': other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': other_lr},

            {"params": [p for n, p in encoder_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay, 'lr': encoder_lr},
            {"params": [p for n, p in encoder_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': encoder_lr},

            {"params": [p for n, p in decoder_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay, 'lr': decoder_lr},
            {"params": [p for n, p in decoder_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': decoder_lr},

        ]

        if user_args.png:
            png_parameters = [
                {"params": [p for n, p in pointer_param_optimizer if not any(nd in n for nd in no_decay)],
                 "weight_decay": weight_decay, 'lr': pointer_lr},
                {"params": [p for n, p in pointer_param_optimizer if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0, 'lr': pointer_lr}
            ]
            optimizer_grouped_parameters.extend(png_parameters)

    if mi_estimator is not None:
        mi_estimator_parameters = [
            {
                "params": list(mi_estimator.parameters()),
                "weight_decay": weight_decay,
            }
        ]
        optimizer_grouped_parameters.extend(mi_estimator_parameters)

    if user_args.novograd:
        print('\n>>> Use NovoGrad Optimizer .')
        optimizer = NovoGrad(optimizer_grouped_parameters, lr=5e-3,
                             weight_decay=1e-3, eps=eps)
    else:
        print('\n>>> Use AdamW Optimizer .')
        optimizer = AdamW(optimizer_grouped_parameters, lr=opt_lr,
                          weight_decay=weight_decay, eps=eps)

    if lka:
        optimizer = Lookahead(optimizer, 0.5, 5)

    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

    return optimizer, scheduler


def train():

    if user_args.lbs:
        lbs_loss_fct = label_smoothing_loss(ignore_index=-100,
                                            epsilon=user_args.smoothing)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    tokenizer = BertTokenizerFast.from_pretrained(user_args.pretrain_model_path)
    model = build_model()

    model.config.decoder_start_token_id = tokenizer.sep_token_id
    model.config.forced_eos_token_id = tokenizer.sep_token_id
    model.config.eos_token_id = tokenizer.sep_token_id

    data_dict = read_data(tokenizer)
    dataset = GAIICDataset(data_dict)

    collator = GAIICDataCollator(tokenizer)

    train_data_loader = DataLoader(
        dataset,  # train_dataset, dataset
        batch_size=user_args.batch_size,
        num_workers=user_args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )

    # step setting
    total_steps = len(train_data_loader) * user_args.total_epoch
    warmup_steps = int(user_args.warmup_ratio * total_steps) + 1
    # warmup_steps = 1

    ema_start_steps = 1
    adv_start_steps = 1
    rdrop_start_steps = 1
    label_smoothing_start_steps = 1

    optimizer, scheduler = build_optimizer(model,
                                           total_steps,
                                           warmup_steps,
                                           lka=user_args.lka,
                                           layer_wise_lr=user_args.layer_wise_lr, up=user_args.up,
                                           scheduler_type=user_args.scheduler_type)

    global_steps = 0
    cur_avg_loss = 0.
    best_score = 0.
    model.zero_grad()
    for epoch in range(user_args.total_epoch):
        model.train()

        train_iterator = tqdm(train_data_loader,
                              desc=f'Train epoch : {epoch + 1}',
                              total=len(train_data_loader))

        for index, data in enumerate(train_iterator):

            global_steps += 1

            if global_steps == ema_start_steps:
                print('\n>>> EMA starting ... ...')
                if user_args.ema:
                    ema = EMA(model.parameters(), decay=user_args.ema_weight_decay)

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = data['input_ids'], \
                data['attention_mask'], \
                data['decoder_input_ids'], \
                data['decoder_attention_mask'], \
                data['labels']

            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = input_ids.to(device), \
                attention_mask.to(device), \
                decoder_input_ids.to(device), \
                decoder_attention_mask.to(device), \
                labels.to(device)

            if user_args.png:
                output = model(
                    input_ids=input_ids,
                    src=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels
                )
            else:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels
                )
            normal_lm_logits = output.logits

            if user_args.lbs:
                if global_steps == label_smoothing_start_steps:
                    print('label smooth starting ... ...')
                if global_steps >= label_smoothing_start_steps:
                    loss = lbs_loss_fct(normal_lm_logits, labels)
                else:
                    loss = output.loss
            else:
                loss = output.loss

            """
            Regular decoder outputs distribution
            """
            if user_args.rdrop:
                if global_steps >= rdrop_start_steps:
                    second_lm_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels
                    ).logits

                """
                RDrop : KL-Div with sequence level
                """
                if user_args.rdrop:
                    if global_steps == rdrop_start_steps:
                        print('\n>>> Rdrop starting ... ...')
                    if rdrop_start_steps <= global_steps:
                        rdrop_loss = compute_kl_loss(normal_lm_logits, second_lm_logits,
                                                     torch.ne(labels, -100).to(torch.bool).to(device))
                        if user_args.lbs:
                            if global_steps >= warmup_steps:
                                loss1 = lbs_loss_fct(second_lm_logits, labels)
                            else:
                                loss1 = loss_fct(second_lm_logits.view(-1, model.config.vocab_size), labels.view(-1))
                        else:
                            loss1 = loss_fct(second_lm_logits.view(-1, model.config.vocab_size), labels.view(-1))
                        loss = (loss + loss1) * 0.5 + user_args.rdrop_alpha * rdrop_loss

            loss.backward()
            cur_avg_loss += loss.mean().item()

            """
            adv mode : share embeddings attack
            """
            if user_args.adv:
                if global_steps == adv_start_steps:
                    print('\n>>> Adv train starting ... ...')
                """
                fgm
                """
                if user_args.fgm:
                    if global_steps == adv_start_steps:
                        print('>> FGM starting ... ...')
                        fgm = FGM(model, user_args.adv_eps)
                    if global_steps >= adv_start_steps:
                        fgm.attack()
                        if user_args.png:
                            adv_output = model(
                                input_ids=input_ids,
                                src=input_ids,
                                attention_mask=attention_mask,
                                decoder_input_ids=decoder_input_ids,
                                decoder_attention_mask=decoder_attention_mask,
                                labels=labels
                            )
                        else:
                            adv_output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                decoder_input_ids=decoder_input_ids,
                                decoder_attention_mask=decoder_attention_mask,
                                labels=labels
                            )
                        adv_loss = lbs_loss_fct(adv_output.logits, labels)
                        # adv_loss = loss_fct(adv_output.logits.view(-1, model.config.vocab_size),
                        #                     labels.view(-1))
                        adv_loss.backward()
                        fgm.restore()

                """
                pgd
                """
                if user_args.pgd:
                    if global_steps == adv_start_steps:
                        print('>> PGD starting ... ...')
                        pgd = PGD(model, user_args.adv_eps)
                    if global_steps >= adv_start_steps:
                        pgd.backup_grad()
                        for t in range(user_args.adv_k):
                            pgd.attack(is_first_attack=(t == 0))
                            if t != user_args.adv_k - 1:
                                model.zero_grad()
                            else:
                                pgd.restore_grad()
                            adv_output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                decoder_input_ids=decoder_input_ids,
                                decoder_attention_mask=decoder_attention_mask,
                                labels=labels
                            )
                            adv_loss = lbs_loss_fct(adv_output.logits, labels)
                            # adv_loss = loss_fct(adv_output.logits.view(-1, model.config.vocab_size),
                            #                     labels.view(-1))
                            adv_loss.backward()
                        pgd.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if user_args.ema:
                if global_steps >= ema_start_steps:
                    ema.update(model.parameters())

            scheduler.step()
            model.zero_grad()

            train_iterator.set_postfix(a_loss=loss.mean().item(),
                                       b_adv_loss=adv_loss.item(),
                                       c_rdrop_loss=rdrop_loss.item(),
                                       d_weighted_rdrop_loss=user_args.rdrop_alpha * rdrop_loss.item(),
                                       g_global_step=global_steps)

        """
        one epoch end
        """
        epoch_avg_loss = cur_avg_loss / len(train_data_loader)
        cur_avg_loss = 0.
        print(f'\n>>> Epoch {epoch + 1}, average loss : {epoch_avg_loss:.4f}')

        """
        evaluate
        """
        if user_args.ema:
            if global_steps >= ema_start_steps:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

        model.eval()

        cider_score, bleu4_score, total_score = 0, 0, 0

        model.save_pretrained(os.path.join(user_args.save_path,
                                           f'epoch_{epoch + 1}_c_{cider_score:.4f}_b_{bleu4_score:.4f}_t_{total_score:.4f}'))
        tokenizer.save_pretrained(os.path.join(user_args.save_path,
                                               f'epoch_{epoch + 1}_c_{cider_score:.4f}_b_{bleu4_score:.4f}_t_{total_score:.4f}'))

        if user_args.ema:
            if global_steps >= ema_start_steps:
                ema.restore(model.parameters())

        model.train()

    print('>>> train completed !!!')


def array2str(arr):
    out = ''
    for i in range(len(arr)):
        if arr[i] == '[PAD]' or arr[i] == '[SEP]':
            break
        if arr[i] == '[CLS]':
            continue
        out += str(arr[i]) + ' '
    if len(out.strip()) == 0:
        out = '0'
    return out.strip()


def clean(decoder_output):
    bad_tokens = ['[CLS]', '[SEP]', '[PAD]']
    new_decoder_output = []
    for token in decoder_output:
        new_decoder_output.append(token.replace(' ', ''))
    new_decoder_output1 = []
    for token in new_decoder_output:
        if token not in bad_tokens:
            new_decoder_output1.append(token)
    return new_decoder_output1


def clean_labels(input_labels, pad_token_id):
    new_labels = []
    for token in input_labels:
        new_labels.append(int(str(token).replace('-100', str(pad_token_id))))
    return new_labels


def prepare_inputs(inputs, device='cuda'):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def evaluate(model, tokenizer, loader, n=-1, accelerator=None):
    metrics = Smoother(100)
    res, gts = [], {}
    tot = 0

    if accelerator is not None:
        # to Accelerator
        model, tokenizer, loader = accelerator.prepare(model, tokenizer, loader)

    all_predictions, all_labels = [], []

    valid_iterator = tqdm(loader, desc='Evaluation', total=len(loader))
    for index, data in enumerate(valid_iterator):

        if 0 < n < tot:
            break

        model_inputs = {
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask']
        }

        labels = data['labels'].to(device)

        if user_args.png:
            model_inputs['src'] = model_inputs['input_ids']

        model_inputs = prepare_inputs(model_inputs, device)

        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                num_beams=user_args.num_beams,
                min_length=user_args.min_length,
                max_length=user_args.generate_max_length,
                early_stopping=user_args.early_stopping,
                no_repeat_ngram_size=user_args.no_repeat_ngram_size,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.cls_token_id,
                eos_token_id=tokenizer.sep_token_id,
                decoder_start_token_id=tokenizer.sep_token_id
            )

        batch_predictions, batch_labels = [], []
        for i in range(len(outputs)):
            decoded_preds = tokenizer.batch_decode(
                outputs[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded_preds = clean(decoded_preds)
            decoded_labels = tokenizer.batch_decode(
                clean_labels(labels[i].cpu().numpy().tolist(), tokenizer.pad_token_id),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded_labels = clean(decoded_labels)

            batch_predictions.append(decoded_preds)
            batch_labels.append(decoded_labels)

        all_predictions.extend(batch_predictions)
        all_labels.extend(batch_labels)

        for i in range(len(batch_predictions)):
            res.append({'image_id': tot, 'caption': [array2str(batch_predictions[i])]})
            gts[tot] = [array2str(batch_labels[i])]
            tot += 1

    CiderD_scorer = CiderD(df='corpus', sigma=15)
    cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
    metrics.update(cider=cider_score)

    all_predictions = [array2str(item) for item in all_predictions]
    all_labels = [array2str(item) for item in all_labels]

    to_calc = {i: [t] for i, t in enumerate(all_labels)}, {i: [p] for i, p in enumerate(all_predictions)}

    bleu4_scorer = Bleu(n=4)
    bleu4_scores, _ = bleu4_scorer.compute_score(*to_calc)
    bleu4_score = bleu4_scores[-1]

    cider_score = metrics.value()['cider']

    score = (2 * cider_score + bleu4_score) / 3
    print(f'\n>>> Cider score : {cider_score:.4f}, Bleu4 score : {bleu4_score:.4f},'
          f' Avg score : {score:.4f}')

    return cider_score, bleu4_score, score


def main():
    seed_everything(user_args.seed)
    print('\n>>> Start training !!!\n')
    train()


if __name__ == '__main__':
    main()
