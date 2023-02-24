# 模型推理

以配置文件 [resume.yaml](../../examples/bert_crf/configs/resume.yaml) 为例，AdaSeq模型训练完成后，可用于推理的的模型文件和相关配置会存放于 `./experiments/resume/${RUN_NAME}/output`(modelscope>=1.3.0版本后目录变为`output_best`)。

```
./experiments/resume/${RUN_NAME}/output
├── config.json
├── configuration.json
├── pytorch_model.bin
├── special_tokens_map.json  # 可选
├── tokenizer_config.json  # 可选
├── tokenizer.json  # 可选
└── vocab.txt  # 可选
```

### 1. 使用本地模型进行推理

可以使用以下python代码进行模型推理，其中模型路径请使用本地绝对路径。
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    Tasks.named_entity_recognition,
    '/work_dir/experiments/resume/221227191502.310576/output'  # 绝对路径
)
result = p('1984年出生，中国国籍，汉族，硕士学历')

print(result)
# {'output': [{'type': 'CONT', 'start': 8, 'end': 12, 'span': '中国国籍'}, {'type': 'RACE', 'start': 13, 'end': 15, 'span': '汉族'}, {'type': 'EDU', 'start': 16, 'end': 20, 'span': '硕士学历'}]}
```

### 2. 使用ModelScope模型进行推理

> 保存的模型可以直接发布到ModelScope，参考 [模型发布到ModelScope](./uploading_to_modelscope_zh.md)

可以使用以下python代码进行模型推理，其中模型路径请使用创建的model_id。

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    Tasks.named_entity_recognition,
    'damo/nlp_raner_named-entity-recognition_chinese-base-resume'  # model_id
)
result = p('1984年出生，中国国籍，汉族，硕士学历')

print(result)
# {'output': [{'type': 'CONT', 'start': 8, 'end': 12, 'span': '中国国籍'}, {'type': 'RACE', 'start': 13, 'end': 15, 'span': '汉族'}, {'type': 'EDU', 'start': 16, 'end': 20, 'span': '硕士学历'}]}
```
