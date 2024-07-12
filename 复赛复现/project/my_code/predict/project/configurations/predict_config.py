# coding:utf-8


class Config(object):

    finetuned_model_path1 = './finetune_model/bart_finetuned_model_swa_base_ade_output'
    finetuned_model_path2 = './finetune_model/bart_finetuned_model_swa_base_n_gram_output'

    finetuned_model_path3 = './finetune_model/half_bart_finetuned_model_swa_large_ade_eps0_15_output'
    finetuned_model_path4 = './finetune_model/half_bart_finetuned_model_swa_large_n-gram_output'

    finetuned_model_path5 = './finetune_model/half_randeng_finetuned_model_swa_ade_output'
    finetuned_model_path6 = './finetune_model/half_randeng_finetuned_model_swa_gsg_output'

    finetuned_model_path7 = './finetune_model/cpt_finetuned_model_swa_base_dae_output'
    finetuned_model_path8 = './finetune_model/cpt_finetuned_model_swa_base_n_gram_output'
    finetuned_model_path9 = './finetune_model/half_cpt_finetuned_model_swa_large-n-gram-eps0_15_output'

    batch_size = 24  # 32
    num_workers = 0

    min_length = 2
    max_length = 200
    num_beams = 3
    no_repeat_ngram_size = 8
    early_stopping = True

    device = 'cuda'
    seed = 42

