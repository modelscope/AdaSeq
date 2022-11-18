# Partial Bert-CRF
In many scenarios, named entity recognition (NER) models severely suffer from unlabeled entity problem, where the entities of a sentence may not be fully annotated. Partial CRF is a parameter estimation method for Conditional Random Fields (CRFs), which enables us to use such incomplete annotations [(Tsuboi et al.)](https://aclanthology.org/C08-1113/).

## data preprocessing
Partially annotated entity/span should be marked as B-P.
You can process the trainset and set the path to `dataset: data_files: train` of `configs/msra.yml`.

## Training a new model
```
python -m scripts.train -c examples/partial_bert_crf/configs/resume.yaml
```

## Benchmarks
TODO
