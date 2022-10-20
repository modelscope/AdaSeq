# 简介
实体理解统一框架，集成实体理解各任务SOTA模型，支持模型的快速开发、调试和部署。


# 使用方法

## 1. 依赖安装
```
pip install -r requirements.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

## 2. 模型训练
```
python scripts/train.py -c examples/bert_crf/configs/resume.yaml -t ner-trainer --seed 0
```

## 3. 模型测试
```
python scripts/test.py -c examples/bert_crf/configs/resume.yaml -t ner-trainer -cp checkpoint_path
```

# 开发文档
1. [UNER开发规约](https://yuque.antfin-inc.com/docs/share/7088e485-5817-4beb-8a28-f8de7dd95a9a?# 《UNER开发规约》)
2. [Modelscope文档中心](https://modelscope.cn/docs/%E9%A6%96%E9%A1%B5)
