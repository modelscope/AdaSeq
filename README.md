# 简介
实体理解统一框架，集成实体理解各任务SOTA模型，支持模型的快速开发、调试和部署。


# 使用方法

## 1. 依赖安装
```
pip install -r requirements.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

## 2. 模型训练
```
python -m examples.train -c examples/bert_crf/configs/resume.yaml --seed 0
```


# 开发文档
1. [Python开发规约](https://yuque.antfin-inc.com/docs/share/d31e3bbe-9d95-44b4-8fb9-e232cdb083c7?# 《Python开发规约》)
2. [Modelscope文档中心](https://modelscope.cn/docs/%E9%A6%96%E9%A1%B5)
