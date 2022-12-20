# 复现论文实验结果
AdaSeq提供了很多序列理解相关论文的代码复现和实验参数。用户可以简单的通过一条命令来复现论文。

比如说，复现 [BABERT](../../examples/babert)，你可以运行：
```commandline
python scripts/train.py -c examples/babert/configs/cws/pku.yaml
```

*注意*：由于PyTorch版本的差异或者AdaSeq版本的差异，复现出的结果可能会在小范围内发生波动。

最后，我们正在持续性地复现领域中更多的经典论文和SOTA论文。我们也欢迎社区用户参与进来，一起丰富AdaSeq。
