 set -x
set -e

# PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# cd ${PROJECT_DIR}
# 需要提前将预训练模型放到 ./pretrain/pretrain_model/ 下

ls ${PROJECT_DIR}
# 准备数据 ok
python pretrain/vocab/prepare_data.py
# 得到vocab ok
python pretrain/vocab/build_tokenizer.py

# 得到 精简vocab的bart-base模型  ok
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/bart-base-chinese \
    --save_path ./pretrain/pretrain_model/new-bart-base \
    --model_name bart
# 得到 精简vocab的bart-large模型  ok
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/bart-large-chinese \
    --save_path ./pretrain/pretrain_model/new-bart-large \
    --model_name bart
# 得到 精简vocab的cpt-base模型  ok
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/cpt-base \
    --save_path ./pretrain/pretrain_model/new-cpt-base \
    --model_name cpt
# 得到 精简vocab的cpt-large模型  ok
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/cpt-large \
    --save_path ./pretrain/pretrain_model/new-cpt-large \
    --model_name cpt
# 得到 精简vocab的pegasus模型  ok
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/Randeng-Pegasus-238M-Chinese \
    --save_path ./pretrain/pretrain_model/new-pegasus-base \
    --model_name pegasus

# 预训练 bart base dae
python pretrain/run_pretrain_bart_dae.py --total_epoch 120 \
    --batch_size 128 \
    --pretrain_model_path ./pretrain/pretrain_model/new-bart-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-bart-base-dae
# 预训练 bart large dae
python pretrain/run_pretrain_bart_dae.py --total_epoch 60 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-bart-large \
    --save_path ./pretrain/pretrain_model/post-pretrained-bart-large-dae
# 预训练 bart base n-gram
python pretrain/run_pretrain_bart_n_gram.py --total_epoch 120 \
    --batch_size 128 \
    --pretrain_model_path ./pretrain/pretrain_model/new-bart-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-bart-base-n-gram
# 预训练 bart large n-gram
python pretrain/run_pretrain_bart_dae.py --total_epoch 60 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-bart-large \
    --save_path ./pretrain/pretrain_model/post-pretrained-bart-large-n-gram

# 预训练 cpt base dae
python pretrain/run_pretrain_cpt_dae.py --total_epoch 120 \
    --batch_size 128 \
    --pretrain_model_path ./pretrain/pretrain_model/new-cpt-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-cpt-base-dae
# 预训练 cpt large dae
python pretrain/run_pretrain_cpt_dae.py --total_epoch 60 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-cpt-large \
    --save_path ./pretrain/pretrain_model/post-pretrained-cpt-large-dae
# 预训练 cpt base n-gram
python pretrain/run_pretrain_cpt_n_gram.py --total_epoch 120 \
    --batch_size 128 \
    --pretrain_model_path ./pretrain/pretrain_model/new-cpt-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-cpt-base-n-gram
# 预训练 cpt large n-gram
python pretrain/run_pretrain_cpt_dae.py --total_epoch 60 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-cpt-large \
    --save_path ./pretrain/pretrain_model/post-pretrained-cpt-large-n-gram

# 预训练 pegasus base dae
python pretrain/run_pretrain_pegasus_dae.py --total_epoch 120 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-pegasus-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-dae
# 预训练 pegasus base gsg
python pretrain/run_pretrain_pegasus_gsg.py --total_epoch 120 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-pegasus-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-gsg

# 微调 bart base dae
python finetune/run_finetune.py --seed 42 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/bart_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-base-dae/epoch_120 \
    --model_name bart
# 微调 bart base n-gram
python finetune/run_finetune.py --seed 2023 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/bart_finetuned_model_base_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-base-n-gram/epoch_120 \
    --model_name bart
# 微调 bart large dae
python finetune/run_finetune.py --seed 1314 \
    --smoothing 0.15 \
    --save_path ./finetune/finetune_models/bart_finetuned_model_large_dae_eps0_15 \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-large-dae/epoch_60 \
    --model_name bart
# 微调 bart large n-gram
python finetune/run_finetune.py --seed 2023 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/bart_finetuned_model_large_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-large-n-gram/epoch_120 \
    --model_name bart

# 微调 cpt base dae
python finetune/run_finetune.py --seed 42 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/cpt_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-base-dae/epoch_120 \
    --model_name cpt
# 微调 cpt base n-gram
python finetune/run_finetune.py --seed 2023 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/cpt_finetuned_model_base_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-base-n-gram/epoch_120 \
    --model_name cpt
# 微调 cpt large n-gram
python finetune/run_finetune.py --seed 123456 \
    --smoothing 0.15 \
    --save_path ./finetune/finetune_models/cpt_finetuned_model_large_n_gram_eps0_15 \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-large-n-gram/epoch_60 \
    --model_name cpt

# 微调 pegasus base dae
python finetune/run_finetune.py --seed 42 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/pegasus_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-dae/epoch_120 \
    --model_name pegasus
# 微调 pegasus base gsg
python finetune/run_finetune.py --seed 2023 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/pegasus_finetuned_model_base_gsg \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-gsg/epoch_120 \
    --model_name pegasus

# swa bart base dae   ../best_model
python finetune/swa_fintune.py --seed 42 \
    --model_name bart \
    --save_path ../best_model/bart_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-base-dae/epoch_120 \
    --swa_output_dir ./predict/project/finetune_model/bart_finetuned_model_swa_base_ade_output
# swa bart base n-gram
python finetune/swa_fintune.py --seed 2023 \
    --model_name bart \
    --save_path ../best_model/bart_finetuned_model_base_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-base-n-gram/epoch_120 \
    --swa_output_dir ./predict/project/finetune_model/bart_finetuned_model_swa_base_n_gram_output
# swa bart large dae
python finetune/swa_fintune.py --seed 1314 \
    --model_name bart \
    --save_path ./finetune/finetune_models/bart_finetuned_model_large_dae_eps0_15 \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-large-dae/epoch_60 \
    --swa_output_dir ./finetune/swa_finetune_models/bart_finetuned_model_swa_large_dae_eps0_15_output
# swa bart large n-gram
python finetune/swa_fintune.py --seed 2023 \
    --model_name bart \
    --save_path ./finetune/finetune_models/bart_finetuned_model_large_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-large-n-gram/epoch_120 \
    --swa_output_dir ./finetune/swa_finetune_models/bart_finetuned_model_swa_large_n_gram_output

# swa cpt base dae
python finetune/swa_fintune.py --seed 42 \
    --model_name cpt \
    --save_path ../best_model/cpt_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-base-dae/epoch_120 \
    --swa_output_dir ./predict/project/finetune_model/cpt_finetuned_model_swa_base_dae_output
# swa cpt base n-gram
python finetune/swa_fintune.py --seed 2023 \
    --model_name cpt \
    --save_path ../best_model/cpt_finetuned_model_base_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-base-n-gram/epoch_120 \
    --swa_output_dir ./predict/project/finetune_model/cpt_finetuned_model_swa_base_n_gram_output
# swa cpt large n-gram
python finetune/swa_fintune.py --seed 123456 \
    --model_name cpt \
    --save_path ./finetune/finetune_models/cpt_finetuned_model_large_n_gram_eps0_15 \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-large-n-gram/epoch_60 \
    --swa_output_dir ./finetune/swa_finetune_models/cpt_finetuned_model_swa_large_n_gram_eps0_15_output

# swa pegasus base dae
python finetune/swa_fintune.py --seed 42 \
    --model_name pegasus \
    --save_path ./finetune/finetune_models/pegasus_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-dae/epoch_120 \
    --swa_output_dir ./finetune/swa_finetune_models/pegasus_finetuned_model_swa_dae_output
# swa pegasus base gsg
python finetune/swa_fintune.py --seed 2023 \
    --model_name pegasus \
    --save_path ./finetune/finetune_models/pegasus_finetuned_model_base_gsg \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-gsg/epoch_120 \
    --swa_output_dir ./finetune/swa_finetune_models/pegasus_finetuned_model_swa_gsg_output

# half large models
python predict/half_models.py

# predict
# 打包 predict 下面的 project文件夹
