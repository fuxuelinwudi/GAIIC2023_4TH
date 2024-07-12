# coding:utf-8
import os


class Config(object):
    train_path = './data/train.csv'
    semi_train_path = './data/semi_train.csv'
    test_a_path = './data/preliminary_a_test.csv'
    test_b_path = './data/preliminary_b_test.csv'
    pretrain_model_path = './pretrain_model/new-bart-base'
    save_path = './pretrain_model/post-pretrained-bart-ade'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    total_epoch = 120
    batch_size = 256
    encoder_max_length = 199  # 156
    num_workers = 0
    learning_rate = 1e-4
    weight_decay = 1e-2
    warmup_ratio = 1e-1
    eps = 1e-8
    ema = False
    masked_lm_prob = 0.3
    pretrain_with_dev = False

    device = 'cuda'
    seed = 2023





