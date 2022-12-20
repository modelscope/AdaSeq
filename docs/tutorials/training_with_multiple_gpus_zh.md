# 使用多GPU训练

这部分的教程介绍如何使用多块GPU训练模型。

## 单机多卡
```shell
python -m torch.distributed.launch --nproc_per_node=${NUMBER_GPUS} --master_port=${MASTER_PORT} scripts/train.py -c ${cfg_file}
```
- **nproc_per_node** (`int`): 当前主机创建的进程数（使用的GPU个数）， 例如 `--nproc_per_node=8`。
- **master_port** (`int`): 主节点的端口号，例如 `--master_port=29527`。

## 多机多卡
比如说，我们有两个节点（机器）：

Node 1:
```shell
python -m torch.distributed.launch --nproc_per_node=${NUMBER_GPUS} --nnodes=2 --node_rank=0 --master_addr=${MASTER_IP_ADDRESS} --master_port=${MASTER_PORT} scripts/train.py -c ${cfg_file}
```

Node 2:
```shell
python -m torch.distributed.launch --nproc_per_node=${NUMBER_GPUS} --nnodes=2 --node_rank=1 --master_addr=${MASTER_IP_ADDRESS} --master_port=${MASTER_PORT} scripts/train.py -c ${cfg_file}
```
- **nproc_per_node**(`int`): 当前主机创建的进程数（使用的GPU个数）， 例如 `--nproc_per_node=8`。
- **nnodes**(`int`): 节点的个数。
- **node_rank**(`int`): 当前节点的索引值，从0开始。
- **master_addr**(`int`): 主节点的ip地址，例如 `--master_addr=192.168.1.1`。
- **master_port**(`int`): 主节点的端口号，例如 `--master_port=29527`。

## [可选] 修改配置文件
如果运行中发生RuntimeError，可以尝试加入下面这段代码到你的配置文件中。
```yaml
parallel:
  type: DistributedDataParallel
  find_unused_parameters: true
```
