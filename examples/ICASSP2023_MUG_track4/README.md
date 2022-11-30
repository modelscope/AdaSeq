# ICASSP2023 MUG Challenge Track4 Keyphrase Extraction Baseline

This tutorial shows how to train a baseline model for the Keyphrase Extraction track in the ICASSP2023 MUG Challenge.

## 4 Steps to Train a Model

We model KPE as a sequence-labeling problem and apply the Bert-CRF model implemented in [AdaSeq](https://github.com/modelscope/adaseq/examples/ICASSP2023_MUG_track4/README.md) to solve it.

> AdaSeq is a an easy-to-use library, built on ModelScope, which provides plenty of cutting-edge models, training methods and useful toolkits for sequence understanding tasks.

#### Step1: Requirements & Installation
Python version >= 3.7
```
git clone https://github.com/modelscope/adaseq.git
pip install -r requirements.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

#### Step2: Download & Preprocess Data
1. Download and put `train.json` `dev.json` to `${root_dir}/examples/ICASSP2023_MUG_track4/dataset/`. `${root_dir}` is the absolute path of the AdaSeq repository.

2. Preprocess the downloaded data by splitting it into splits of 128 characters (or longer) and reformat data into CoNLL format.
```
cd examples/ICASSP2023_MUG_track4
python preprocess.py
```

#### Step3: Start Training
1. Modify the config file
Change the `${root_dir}` in `configs/bert_crf_sbert.yaml` to your own ABSOLUTE path of this repo.

2. Start training
Let’s train a Bert-CRF model using the preset script as an example.
```
cd ${root_dir}
python scripts/train.py -c examples/ICASSP2023_MUG_track4/configs/bert_crf_sbert.yaml
```

Also, there are many methods other than Bert-CRF implemented in AdaSeq, such as RaNER, Global-Pointer, etc.

What's more, a [grid search tool](https://github.com/modelscope/adaseq/docs/tutorials/hyperparameter_tuning_with_grid_search.md) is provided for efficient tuning.

#### Step4: Evaluate Your Model
First install some requirements in the evaluation script.
```
pip install jieba rouge yake
```

Use `evaluate_kw.py` to evaluate the predictions. For KPE, the union of the labels from the three annotators is used as the final manual labels for training and evaluation.

There are 4 necessary parameters:
- `<data_path>`: the source test data. e.g. dev.json.
- `<pred_path>`: the prediction file output by your model.
- `<doc_split_path`>:  the file recording where documents are split, which is output by preprocessing.
- `<out_path>`: the file path in which the evaluation log will be.

For example:

```shell
cd examples/ICASSP2023_MUG_track4
python evaluate_kw.py dataset/dev.json ${root_dir}/experiments/kpe_sbert/outputs/${datetime}/pred.txt dataset/split_list_dev.json evaluation.log
```

## Benchmarks

| Split |  Model   |               Backbone                | Exact/Partial F1 @10 | Exact/Partial F1 @15 | Exact/Partial F1 @20 |                                                           Checkpoint                                                            |
|:-----:|:--------:|:-------------------------------------:|:--------------------:|:--------------------:|:--------------------:|:-------------------------------------------------------------------------------------------------------------------------------:|
|  Dev  |   yake   |                   -                   |      15.0/24.3       |      19.8/30.4       |      20.4/32.1       |                                                                -                                                                |
|  Dev  | Bert-CRF |         sijunhe/nezha-cn-base         |      35.6/43.2       |      38.1/49.5       |      37.2/48.1       |                                                                -                                                                |
|  Dev  | Bert-CRF | damo/nlp_structbert_backbone_base_std |      35.9/47.7       |      40.1/52.2       |      39.4/51.1       | [ModelScope](https://modelscope.cn/models/damo/nlp_structbert_keyphrase-extraction_base-icassp2023-mug-track4-baseline/summary) |
