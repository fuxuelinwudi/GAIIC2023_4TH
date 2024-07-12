# coding:utf-8


import os
import copy
import torch
from transformers import BartForConditionalGeneration, BertTokenizerFast, BartConfig
from transformers import PegasusForConditionalGeneration, PegasusConfig
from modeling.modeling_cpt import CPTForConditionalGeneration, CPTConfig


def to_half(model_path, save_path, model_type):
    print('*' * 50)
    print(model_path)
    print(save_path)
    print('*'*50)

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    tokenizer.save_pretrained(save_path)

    if model_type == 'bart':
        config = BartConfig.from_pretrained(model_path)
        config.save_pretrained(save_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
    elif model_type == 'cpt':
        config = CPTConfig.from_pretrained(model_path)
        config.save_pretrained(save_path)
        model = CPTForConditionalGeneration.from_pretrained(model_path)
    else:
        config = PegasusConfig.from_pretrained(model_path)
        config.save_pretrained(save_path)
        model = PegasusForConditionalGeneration.from_pretrained(model_path)

    model.half()

    model_state_dict = model.state_dict()
    torch.save(model_state_dict, os.path.join(save_path, 'pytorch_model.bin'))


bart_model_path_list = ['./finetune/swa_finetune_models/bart_finetuned_model_swa_large_dae_eps0_15_output',
                        './finetune/swa_finetune_models/bart_finetuned_model_swa_large_n_gram_output']

pegasus_model_path_list = ['./finetune/swa_finetune_models/pegasus_finetuned_model_swa_dae_output',
                           './finetune/swa_finetune_models/pegasus_finetuned_model_swa_gsg_output']

cpt_model_path_list = ['./finetune/swa_finetune_models/cpt_finetuned_model_swa_large_n_gram_eps0_15_output']

save_path_list = ['../../best_model/half_bart_finetuned_model_swa_large_ade_eps0_15_output',
                  '../../best_model/half_bart_finetuned_model_swa_large_n-gram_output',

                  '../../best_model/half_randeng_finetuned_model_swa_ade_output',
                  '../../best_model/half_randeng_finetuned_model_swa_gsg_output',

                  '../../best_model/half_cpt_finetuned_model_swa_large-n-gram-eps0_15_output']


to_half(bart_model_path_list[0], save_path_list[0], 'bart')
to_half(bart_model_path_list[1], save_path_list[1], 'bart')

to_half(pegasus_model_path_list[0], save_path_list[2], 'pegasus')
to_half(pegasus_model_path_list[1], save_path_list[3], 'pegasus')

to_half(cpt_model_path_list[0], save_path_list[4], 'cpt')


