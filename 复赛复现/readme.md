### 柠檬草的味道团队复赛复现说明文档

# 代码说明





## 环境配置

我们团队在原有官方提供的“gaiic 比赛推断-基础镜像”基础上自定义了自己队伍的镜像，镜像名称为“柠檬草的味道复赛镜像”，安装包包括：

```python
transformers==4.29.2
numpy==1.24.3
scikit-learn==1.2.2
rouge==1.0.1
accelerate==0.19.0
joblib==1.2.0
nltk==3.8.1
pandas==2.0.1
six==1.16.0
tqdm==4.65.0
pycocoevalcap==1.2
```

##  预训练语言模型

1、使用了复旦大学开源的 cpt-large 模型，[模型权重获取链接](https://huggingface.co/fnlp/cpt-large/tree/main)和cpt-base模型，[模型权重获取链接](https://huggingface.co/fnlp/cpt-base/tree/main)

2、使用了复旦大学开源的bart-large-chinese模型，[模型权重获取链接](https://huggingface.co/fnlp/bart-large-chinese)和cpt-base-chinese模型，[模型权重获取链接](https://huggingface.co/fnlp/bart-large-chinese/tree/main)

3、使用了IDEA研究院开源的Randeng-Pegasus-238M-Chinese，[模型权重获取链接](https://huggingface.co/IDEA-CCNL/Randeng-Pegasus-238M-Chinese)

对project/my_code/modeling文件夹下的modeling_bart.py、modeling_cpt,py、modeling_pagasus.py中的BartConditionalGeneration 、CPTForConditionalGeneration 和PegasusForConditionalGeneration初始化。

## 算法

整体思路为：预训练+微调形式，然后融合

### 整体思路介绍

一、利用比赛过程中的初赛训练集、A/B榜测试集以及复赛训练集的数据，我们队伍对此次赛题数据进行适应性的预训练，针对预训练任务，主要采用：ngram-mask、借鉴了BART模型的DAE预训练任务和IDEA研究院提出的封神榜系列模型中Pegasus的自监督Gap Sentences Generation预训练任务。

二、针对上述提及的三个预训练任务和三种类型的生成式模型，我们队伍结合此次数据集预训练了如下几种模型：bart base dae、bart large dae、bart base n-gram、bart large n-gram、cpt base dae、cpt large dae、cpt base n-gram、cpt large n-gram、pegasus base dae、pegasus base gsg。

三、通过预训练来适应此次脱敏数据集，然后将预训练得到的模型权重对初赛训练集和复赛训练集进行微调，最终选择对如下几种预训练之后的模型进行微调：bart base dae、bart base n-gram、bart large dae、bart large n-gram、cpt base dae、cpt base n-gram、cpt large n-gram、pegasus base dae、pegasus base gsg。

四、微调的迭代epoch为10，我们队伍对第6-10轮保存的模型进行随机权重平均方法(SWA)，让模型达到更好的收敛效果。

五、微调保存的9个模型，考虑到复赛后台运行的显存大小以及上传文件内容大小的问题，我们对5个模型采用半精度保存，分别是bart large dae、bart large ngram、cpt large ngram、Pegasus base dae和pegasus base gsg。

六、对上述九种模型进行微调之后，然后通过融合的方式对测试集进行预测，最后得到我们队伍在复赛B榜上的成绩。

### 方法的创新点

1. 首先针对脱敏数据，构建相应的词表。因为本题为生成类型赛题，需要考虑到分词器是否会将单词分解，以至于在模型解码过程中一些 id ，如1030，1031等会出现被分解为10,30,31这样的字词问题。因此，通过自己重建词表（不使用 SentencePiece）以解决这样一个问题；
2. 通过构建好的词表，将原始预训练模型的权重中对应单词的 embedding 取出，并赋予到新的初始化的 embedding 中，同时对 oov 的 embedding 进行原始预训练模型的权重的随机选取，以加速模型在脱敏数据上的训练收敛速度；
3. 引用模型集成策略，给予微调模型不同的权重比，不同模型的权重如下所示bart base dae=0.225、bart base n-gram=0.225、bart large dae=0.125、bart large n-gram=0.125、pegasus dae=0.15、pegasus gsg=0.15、cpt base dae=0.35、cpt base n-gram=0.35、cpt large=0.3，最后得到多个模型的融合结果；
4. 预训练采用 ngram、dae、gsg 的预训练方式，对脱敏数据进行领域预训练，使模型能够适应任务数据；
5. 在微调过程中使用 label smooth 策略缓解模型在训练过程中的过拟合现象，并且针对不同模型使用label smooth时引入两个不同参数0.12和0.15去控制smooth的比例；
6. 使用传统的一些 tricks 来提升模型的性能，如 FGM，SWA，EMA等；

## 训练流程

模型训练过程分为了如下几部分代码，主要是预训练+微调两大部分，代码介绍如下所示。

1、预训练任务中，我们队伍利用初赛训练集、A/B榜测试集以及复赛训练集的数据构建预训练数据，见代码1；

2、然后根据预训练数据构建模型用到的统一的字典vocab.txt用于五种类型的预训练模型，见代码2；

3、通过构建好的词表，将原始预训练模型的权重中对应单词的 embedding 取出构建新的模型，介绍见创新点2，代码为3-7；

4、预训练任务介绍见整体思路介绍模块二中，预训练的代码为8-17；

5、微调任务介绍见整体思路介绍模块三中，微调的代码为18-26；

6、对模型进行随机权重平均方法见整体思路介绍模块四中，代码为27-35

7、对模型采用半精度保存的代码为36

```python
# 1、准备数据
python pretrain/vocab/prepare_data.py
# 2、得到vocab
python pretrain/vocab/build_tokenizer.py

# 3、得到 精简vocab的bart-base模型
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/bart-base-chinese \
    --save_path ./pretrain/pretrain_model/new-bart-base \
    --model_name bart
# 4、得到 精简vocab的bart-large模型
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/bart-large-chinese \
    --save_path ./pretrain/pretrain_model/new-bart-large \
    --model_name bart
# 5、得到 精简vocab的cpt-base模型
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/cpt-base \
    --save_path ./pretrain/pretrain_model/new-cpt-base \
    --model_name cpt
# 6、得到 精简vocab的cpt-large模型
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/cpt-large \
    --save_path ./pretrain/pretrain_model/new-cpt-large \
    --model_name cpt
# 7、得到 精简vocab的pegasus模型
python pretrain/vocab/build_model.py --model_path ../../input/pretrain_mode/Randeng-Pegasus-238M-Chinese \
    --save_path ./pretrain/pretrain_model/new-pegasus-base \
    --model_name pegasus

# 8、预训练 bart base dae
python pretrain/run_pretrain_bart_dae.py --total_epoch 120 \
    --batch_size 128 \
    --pretrain_model_path ./pretrain/pretrain_model/new-bart-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-bart-base-dae
# 9、预训练 bart large dae
python pretrain/run_pretrain_bart_dae.py --total_epoch 60 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-bart-large \
    --save_path ./pretrain/pretrain_model/post-pretrained-bart-large-dae
# 10、预训练 bart base n-gram
python pretrain/run_pretrain_bart_n_gram.py --total_epoch 120 \
    --batch_size 128 \
    --pretrain_model_path ./pretrain/pretrain_model/new-bart-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-bart-base-n-gram
# 11、预训练 bart large n-gram
python pretrain/run_pretrain_bart_dae.py --total_epoch 60 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-bart-large \
    --save_path ./pretrain/pretrain_model/post-pretrained-bart-large-n-gram

# 12、预训练 cpt base dae
python pretrain/run_pretrain_cpt_dae.py --total_epoch 120 \
    --batch_size 128 \
    --pretrain_model_path ./pretrain/pretrain_model/new-cpt-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-cpt-base-dae
# 13、预训练 cpt large dae
python pretrain/run_pretrain_cpt_dae.py --total_epoch 60 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-cpt-large \
    --save_path ./pretrain/pretrain_model/post-pretrained-cpt-large-dae
# 14、预训练 cpt base n-gram
python pretrain/run_pretrain_cpt_n_gram.py --total_epoch 120 \
    --batch_size 128 \
    --pretrain_model_path ./pretrain/pretrain_model/new-cpt-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-cpt-base-n-gram
# 15、预训练 cpt large n-gram
python pretrain/run_pretrain_cpt_dae.py --total_epoch 60 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-cpt-large \
    --save_path ./pretrain/pretrain_model/post-pretrained-cpt-large-n-gram

# 16、预训练 pegasus base dae
python pretrain/run_pretrain_pegasus_dae.py --total_epoch 120 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-pegasus-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-dae
# 17、预训练 pegasus base gsg
python pretrain/run_pretrain_pegasus_gsg.py --total_epoch 120 \
    --batch_size 32 \
    --pretrain_model_path ./pretrain/pretrain_model/new-pegasus-base \
    --save_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-gsg

# 18、微调 bart base dae
python finetune/run_finetune.py --seed 42 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/bart_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-base-dae/epoch_120 \
    --model_name bart
# 19、微调 bart base n-gram
python finetune/run_finetune.py --seed 2023 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/bart_finetuned_model_base_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-base-n-gram/epoch_120 \
    --model_name bart
# 20、微调 bart large dae
python finetune/run_finetune.py --seed 1314 \
    --smoothing 0.15 \
    --save_path ./finetune/finetune_models/bart_finetuned_model_large_dae_eps0_15 \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-large-dae/epoch_60 \
    --model_name bart
# 21、微调 bart large n-gram
python finetune/run_finetune.py --seed 2023 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/bart_finetuned_model_large_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-large-n-gram/epoch_120 \
    --model_name bart

# 22、微调 cpt base dae
python finetune/run_finetune.py --seed 42 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/cpt_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-base-dae/epoch_120 \
    --model_name cpt
# 23、微调 cpt base n-gram
python finetune/run_finetune.py --seed 2023 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/cpt_finetuned_model_base_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-base-n-gram/epoch_120 \
    --model_name cpt
# 24、微调 cpt large n-gram
python finetune/run_finetune.py --seed 123456 \
    --smoothing 0.15 \
    --save_path ./finetune/finetune_models/cpt_finetuned_model_large_n_gram_eps0_15 \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-large-n-gram/epoch_60 \
    --model_name cpt

# 25、微调 pegasus base dae
python finetune/run_finetune.py --seed 42 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/pegasus_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-dae/epoch_120 \
    --model_name pegasus
# 26、微调 pegasus base gsg
python finetune/run_finetune.py --seed 2023 \
    --smoothing 0.12 \
    --save_path ./finetune/finetune_models/pegasus_finetuned_model_base_gsg \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-gsg/epoch_120 \
    --model_name pegasus

# 27、swa bart base dae   ../best_model
python finetune/swa_fintune.py --seed 42 \
    --model_name bart \
    --save_path ../best_model/bart_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-base-dae/epoch_120 \
    --swa_output_dir ./predict/project/finetune_model/bart_finetuned_model_swa_base_ade_output
# 28、swa bart base n-gram
python finetune/swa_fintune.py --seed 2023 \
    --model_name bart \
    --save_path ../best_model/bart_finetuned_model_base_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-base-n-gram/epoch_120 \
    --swa_output_dir ./predict/project/finetune_model/bart_finetuned_model_swa_base_n_gram_output
# 29、swa bart large dae
python finetune/swa_fintune.py --seed 1314 \
    --model_name bart \
    --save_path ./finetune/finetune_models/bart_finetuned_model_large_dae_eps0_15 \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-large-dae/epoch_60 \
    --swa_output_dir ./finetune/swa_finetune_models/bart_finetuned_model_swa_large_dae_eps0_15_output
# 30、swa bart large n-gram
python finetune/swa_fintune.py --seed 2023 \
    --model_name bart \
    --save_path ./finetune/finetune_models/bart_finetuned_model_large_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-bart-large-n-gram/epoch_120 \
    --swa_output_dir ./finetune/swa_finetune_models/bart_finetuned_model_swa_large_n_gram_output

# 31、swa cpt base dae
python finetune/swa_fintune.py --seed 42 \
    --model_name cpt \
    --save_path ../best_model/cpt_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-base-dae/epoch_120 \
    --swa_output_dir ./predict/project/finetune_model/cpt_finetuned_model_swa_base_dae_output
# 32、swa cpt base n-gram
python finetune/swa_fintune.py --seed 2023 \
    --model_name cpt \
    --save_path ../best_model/cpt_finetuned_model_base_n_gram \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-base-n-gram/epoch_120 \
    --swa_output_dir ./predict/project/finetune_model/cpt_finetuned_model_swa_base_n_gram_output
# 33、swa cpt large n-gram
python finetune/swa_fintune.py --seed 123456 \
    --model_name cpt \
    --save_path ./finetune/finetune_models/cpt_finetuned_model_large_n_gram_eps0_15 \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-cpt-large-n-gram/epoch_60 \
    --swa_output_dir ./finetune/swa_finetune_models/cpt_finetuned_model_swa_large_n_gram_eps0_15_output

# 34、swa pegasus base dae
python finetune/swa_fintune.py --seed 42 \
    --model_name pegasus \
    --save_path ./finetune/finetune_models/pegasus_finetuned_model_base_dae \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-dae/epoch_120 \
    --swa_output_dir ./finetune/swa_finetune_models/pegasus_finetuned_model_swa_dae_output
# 35、swa pegasus base gsg
python finetune/swa_fintune.py --seed 2023 \
    --model_name pegasus \
    --save_path ./finetune/finetune_models/pegasus_finetuned_model_base_gsg \
    --pretrain_model_path ./pretrain/pretrain_model/post-pretrained-pegasus-base-gsg/epoch_120 \
    --swa_output_dir ./finetune/swa_finetune_models/pegasus_finetuned_model_swa_gsg_output

# 36half large models
python predict/half_models.py

```

## 推理流程

推理过程代码按照主办方要求存于project路径下，运行index.py即可。

## 其他注意事项

