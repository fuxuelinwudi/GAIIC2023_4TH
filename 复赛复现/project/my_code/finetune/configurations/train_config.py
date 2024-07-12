# coding:utf-8

import os


class Config(object):

    train_path = './data/train.csv'
    semi_train_path = './data/semi_train.csv'

    pretrain_model_path = './pretrain_model/bart-pretrain-dae-v4-120'

    save_path = './bart_finetuned_model'

    num_workers = 0
    prefetch = 0

    fp16 = False

    total_epoch = 10
    batch_size = 16
    encoder_max_length = 149 + 48 + 3
    decoder_max_length = 90

    encoder_lr = 5e-5
    decoder_lr = 5e-5
    other_lr = 5e-5
    opt_lr = 5e-5
    weight_decay = 1e-2
    warmup_ratio = 0.1
    eps = 1e-8

    # optimizer choice
    novograd = False
    lka = False

    # scheduler choice
    scheduler_type = 'linear'

    pseudo = False
    pseudo_rate = 0.5

    png = False
    pointer_lr = 1e-4

    layer_wise_lr = False
    up = False

    lbs = True
    smoothing = 0.15  # 0.12

    rdrop = True
    rdrop_alpha = 10000

    ema = True
    ema_weight_decay = 0.999

    adv = True
    fgm = True
    pgd = False
    adv_eps = 0.3  # 1.0
    adv_k = 3

    swa_start = 6
    swa_end = 0
    swa_output_dir = './bart_finetuned_model_swa_output'

    num_beams = 3
    min_length = 4
    generate_max_length = 90
    no_repeat_ngram_size = 8
    early_stopping = True
    use_cache = True

    device = 'cuda'
    seed = 2023  # 2023, 3407, 42

