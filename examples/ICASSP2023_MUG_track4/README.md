# ICASSP2023 MUG Challenge Track4 Keyphrase Extraction Baseline

This example shows how to train a baseline model for the Keyphrase Extraction track in the ICASSP2023 MUG Challenge.

Keyphrase Extraction (KPE) requires extracting top-K keyphrases from a document that reflect its main content. For KPE, the union of the labels from the three annotators is used as the final manual labels for training and evaluation.

We model KPE as a sequence-labeling problem and apply the Bert-CRF model to solve it.


---
## Download competition data
Download the competition data `train.json` and `dev.json` to `${root_dir}/examples/ICASSP2023_MUG_track4/dataset/`

## Preprocessing the src data
Preprocess the src data to the CoNLL format which is applicable to NER.

```
cd examples/ICASSP2023_MUG_track4
python preprocess.py
cd ../../
```
Then change the `${root_dir}` in the configs/bert_crf_sbert.yaml to your own ABSOLUTE path of this repo.

## Training a new Model
Let's train a NER model using the preset script as an example.
```
python scripts/train.py -c examples/ICASSP2023_MUG_track4/configs/bert_crf_sbert.yaml
```

## Evaluating an output prediction
Install some requirements in the evaluation script.
```
pip install jieba rouge yake
```

Use `evaluate_kw.py` to evaluate the predictions.
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


<details>
<summary>See details for training</summary>

### Model-specific Arguments
```
preprocessor:
  tag_scheme: BIOES  # (str, optional): The tag scheme used for sequence-labeling tasks. Possible candidates are [`BIO`, `BIOES`]. Default to `BIOES`.
model:
  word_dropout: 0.1  # (float, optional): Word-level/token-level dropout probability. Default to `0`.
  use_crf: true  # (bool, optional): Whether to use CRF decoder. Default to `true`.
```

### Using Custom Dataset
See [tutorial](../../docs/tutorials/preparing_custom_dataset.md)

</details>

## Benchmarks

| Split |  Model   |               Backbone                | Exact/Partial F1 @10 | Exact/Partial F1 @15 | Exact/Partial F1 @20 |
|:-----:|:--------:|:-------------------------------------:|:--------------------:|:--------------------:|:--------------------:|
|  Dev  |   yake   |                   -                   |      15.0/24.3       |      19.8/30.4       |      20.4/32.1       |
|  Dev  | Bert-CRF |         sijunhe/nezha-cn-base         |      35.6/43.2       |      38.1/49.5       |      37.2/48.1       |
|  Dev  | Bert-CRF | damo/nlp_structbert_backbone_base_std |      35.9/47.7       |      40.1/52.2       |      39.4/51.1       |
