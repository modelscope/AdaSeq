# Retrieval-Augmented Named Entity Recognition

RaNER is a re-implementation of our ACL-IJCNLP 2021 paper: [Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning](https://arxiv.org/pdf/2105.03654.pdf) [[github](https://github.com/Alibaba-NLP/CLNER)]

RaNER is a framework for improving the accuracy of NER models through retrieving external contexts, then use the cooperative learning approach to improve the both input views. It can be illustrated as follows:

![](./resources/model_image.jpg)

## Prepare dataset
After we retrieve the external contexts, we can simply concat them to the original sentences. The label used for the contexts should be `X`.

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

#### MultiCoNER
|    Dataset    | Baseline-F1 | RaNER-F1 |                                                 Modelcard & Demo                                                 |
|:-------------:|:-----------:|:--------:|:----------------------------------------------------------------------------------------------------------------:|
| MultiCoNER-BN |    82.69    |  85.11   | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_bangla-large-generic/summary)  |
| MultiCoNER-DE |    91.71    |   95.0   | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_german-large-generic/summary)  |
| MultiCoNER-EN |    88.70    |  96.59   | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_english-large-generic/summary) |
| MultiCoNER-ES |    86.54    |  94.64   | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_spanish-large-generic/summary) |
| MultiCoNER-FA |    81.85    |  95.97   |  [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_farsi-large-generic/summary)  |
| MultiCoNER-HI |    83.13    |  85.28   |  [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_hindi-large-generic/summary)  |
| MultiCoNER-KO |    86.25    |  95.49   | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_korean-large-generic/summary)  |
| MultiCoNER-NL |    89.92    |  97.28   |  [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_dutch-large-generic/summary)  |
| MultiCoNER-RU |    81.52    |  95.14   | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_russian-large-generic/summary) |
| MultiCoNER-TR |    88.52    |  97.83   | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_turkish-large-generic/summary) |
| MultiCoNER-ZH |    85.43    |  91.44   | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_chinese-large-generic/summary) |
Baseline indicates Transformer-CRF model with the same pretrained backbone.

## Citation
```
@inproceedings{wang2021improving,
    title = "{{Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning}}",
    author={Wang, Xinyu and Jiang, Yong and Bach, Nguyen and Wang, Tao and Huang, Zhongqiang and Huang, Fei and Tu, Kewei},
    booktitle = "{the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (\textbf{ACL-IJCNLP 2021})}",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```
