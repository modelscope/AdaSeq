# Distributed Data Parallel

## single machine & multi gpus
```
python -m torch.distributed.launch --nproc_per_node=${NUMBER_GPUS} --master_port=${MASTER_PORT} -m scripts.train -t ner-trainer -c examples/ddp/ddp.yaml
```
- nproc_per_node：当前主机创建的进程数（使用的GPU个数）， 例如`--nproc_per_node=8`。
- master_port：主节点的端口号，例如`--master_port=29527`。

## multi machines & multi gpus
以两个节点为例

节点1：
```shell
python -m torch.distributed.launch --nproc_per_node=${NUMBER_GPUS} --nnodes=2 --node_rank=0 --master_addr=${YOUR_MASTER_IP_ADDRESS} --master_port=${MASTER_PORT} -m scripts.train -t ner-trainer -c examples/ddp/ddp.yaml
```

节点2：
```shell
python -m torch.distributed.launch --nproc_per_node=${NUMBER_GPUS} --nnodes=2 --node_rank=1 --master_addr=${YOUR_MASTER_IP_ADDRESS} --master_port=${MASTER_PORT} -m scripts.train -t ner-trainer -c examples/ddp/ddp.yaml
```

- nproc_per_node：当前主机创建的进程数（使用的GPU个数）， 例如`--nproc_per_node=8`。
- nnodes：节点的个数。
- node_rank：当前节点的索引值。
- master_addr：主节点的ip地址，例如`--master_addr=104.171.200.62`。
- master_port：主节点的端口号，例如`--master_port=29527`。

## Important notes
配置文件:
- find_unused_parameters: true 允许非必要参数（不参与loss计算），如果为false且包含非必要参数，会报错
