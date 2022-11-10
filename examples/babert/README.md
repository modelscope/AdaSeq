# Unsupervised Boundary-Aware Language Model Pretraining for Chinese Sequence Labeling 

In order to enhance the language model's ability to recognize Chinese boundaries in various sequence labeling tasks, we seek to leverage unsupervised statistical boundary information and propose an architecture to encode the information directly into pre-trained language models, resulting in Boundary-Aware BERT (BABERT). [BABERT](https://arxiv.org/abs/2210.15231). 

The overall architecture of the boundary-aware pre-trained language model:
<div align=center><img src="./resource/babert.png" /></div>

## Experiment results

### Chinese Word Segmentation

| Model      	|  CTB6 	|  MSRA 	|  PKU  	|
|------------	|:-----:	|:-----:	|:-----:	|
| BERT       	| 97.35 	| 98.22 	| 96.26 	|
| BERT-wwm   	| 97.39 	| 98.31 	| 96.51 	|
| ERINE      	| 97.37 	| 95.25 	| 96.30 	|
| ERINE-Gram 	| 97.28 	| 98.27 	| 96.36 	|
| Nezha      	| 97.53 	| 98.61 	| 96.67 	|
| BEBERT     	| 97.45 	| 98.44 	| 96.70 	|

### Chinese Part of Speech

| Model         |  CTB6         |  UD1         |  UD2          |
|------------   |:-----:        |:-----:        |:-----:        |
| BERT          | 94.72         | 95.04         | 94.89  |
| BERT-wwm      | 94.84 | 95.50 | 95.41 |
| ERINE | 94.90 | 95.28 | 95.12 |
| ERINE-Gram| 94.93| 95.26 | 95.16 |
| Nezha | 94.98 | 95.57 | 95.52 |
| BEBERT | 95.05 | 95.65 | 95.54 |

### Chinese Named Entity Recognition

| Model      	| Ontonote4 	| Book9 	|  News 	| Finance 	|
|------------	|:---------:	|:-----:	|:-----:	|---------	|
| BERT       	|   80.98   	| 76.11 	| 79.15 	| 85.31   	|
| BERT-wwm   	|   80.87   	| 76.21 	| 79.26 	| 84.97   	|
| ERINE      	|   80.38   	| 76.56 	| 80.36 	| 86.03   	|
| ERINE-Gram 	|   80.96   	| 77.19 	| 79.96 	| 85.31   	|
| Nezha      	|   81.74   	| 77.03 	| 79.81 	| 85.15   	|
| BEBERT     	|   81.90   	| 76.84 	| 80.27 	| 86.89   	|

### Example of training

- Chinese Word Segmentation
```
CUDA_VISIBLE_DEVICES=7 python -m scripts.train -c examples/babert/configs/cws/msra.yaml -t ner-trainer --seed $seed
```

- Part of Speech
```
CUDA_VISIBLE_DEVICES=7 python -m scripts.train -c examples/babert/configs/pos/ud1.yaml -t ner-trainer --seed $seed
```

- Named Entity Recognition
```
CUDA_VISIBLE_DEVICES=7 python -m scripts.train -c examples/babert/configs/ner/ontonotes4.yaml -t ner-trainer --seed $seed
```

## Citation
```
@article{Jiang2022UnsupervisedBL,
  title={Unsupervised Boundary-Aware Language Model Pretraining for Chinese Sequence Labeling},
  author={Peijie Jiang and Dingkun Long and Yanzhao Zhang and Pengjun Xie and Meishan Zhang and M. Zhang},
  journal={ArXiv},
  year={2022},
  volume={abs/2210.15231}
}
```
