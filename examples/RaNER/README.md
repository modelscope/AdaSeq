# Retrieval-Augmented NER

## Prepare dataset
label used for extra text: X

for example:
```
EU B-ORG
rejects O
German B-MISC
call O
to O
boycott O
British B-MISC
lamb O
. O
<EOS> X
EU X
officials X
sought X
in X
vain X
```

## Training a new model
```
python -m scripts.train -c examples/raner/configs/wnut17.yaml
```

## Benchmarks
| Model | wnut16 | wnut17 | conll03 | conllpp |
| ---- | ---- | ---- | ---- | ---- |
| xlm-roberta-large | | | | |
