# Reproducing Results in Published Papers

We've provided several re-implementations and experiment settings of papers in the area of Sequence Understanding. You can reproduce the results simply using one command.

For example, to reproduce [BABERT](../../examples/babert), you can run:
```commandline
python scripts/train.py -c examples/babert/configs/cws/pku.yaml
```

*Notice*: Due to the differences in PyTorch version or AdaSeq version, the metrics MAY fluctuate within a narrow range.

We are continuously working hard to re-implement more and more classic or SOTA papers in this area. And all contributions are welcome to improve AdaSeq.
