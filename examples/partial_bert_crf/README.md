# Partial Bert-CRF
## data preprocessing
Partially annotated entity/span should be marked as B-P.
You can process the trainset and set the path to `dataset: data_files: train` of `configs/msra.yml`.
 
## Training a new model
```
python -m scripts.train -c examples/partial_bert_crf/configs/resume.yaml
```

## Benchmarks
### Chinese

### English
