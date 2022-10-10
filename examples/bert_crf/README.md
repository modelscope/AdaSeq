# Bert-CRF

## Training a new model
```
python -m scripts.train -c examples/bert_crf/configs/resume.yaml
```

## Benchmarks
### Chinese
| Model | msra | ontonotes | resume | weibo | avg |
| ---- | ---- | ---- | ---- | ---- | ---- |
| bert-base-chinese | | | | | |
| bert-base-wwm-ext | | | | | |
| bert-roberta-wwm-ext | | | | | |
| nezha-cn-base | 95.84 | 81.71 | 95.54 | 70.61 | 85.92 |

### English
| Model | wnut16 | wnut17 | conll03 | conllpp | avg |
| ---- | ---- | ---- | ---- | ---- | ---- |
| bert-base-cased | | | | | |
| roberta-base | | | | | |
| xlm-roberta-base | | | | | |
| bert-large-cased | | | | | |
| roberta-large | | | | | |
| xlm-roberta-large | | | | | |
