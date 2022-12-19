# Training with Multiple GPUs
This part of tutorial shows how you can train models with multiple GPUs.

## single machine & multi gpus
```shell
python -m torch.distributed.launch --nproc_per_node=${NUMBER_GPUS} --master_port=${MASTER_PORT} scripts/train.py -c ${cfg_file}
```
- **nproc_per_node** (`int`): Number of GPUs in the current machine, for example `--nproc_per_node=8`.
- **master_port** (`int`): Master port, for example `--master_port=29527`.

## multi machines & multi gpus
For example, we have 2 nodes.

Node 1:
```shell
python -m torch.distributed.launch --nproc_per_node=${NUMBER_GPUS} --nnodes=2 --node_rank=0 --master_addr=${MASTER_IP_ADDRESS} --master_port=${MASTER_PORT} scripts/train.py -c ${cfg_file}
```

Node 2:
```shell
python -m torch.distributed.launch --nproc_per_node=${NUMBER_GPUS} --nnodes=2 --node_rank=1 --master_addr=${MASTER_IP_ADDRESS} --master_port=${MASTER_PORT} scripts/train.py -c ${cfg_file}
```
- **nproc_per_node**(`int`): Number of GPUs in the current machine, for example `--nproc_per_node=8`.
- **nnodes**(`int`): Number of nodes.
- **node_rank**(`int`): Rank of current node, starting from 0.
- **master_addr**(`int`): Master IP, for example `--master_addr=192.168.1.1`.
- **master_port**(`int`): Master port, for example `--master_port=29527`.

## [Optional] Modify configuration file
Add this script to the end of your configuration file when RuntimeError occurs.
```yaml
parallel:
  type: DistributedDataParallel
  find_unused_parameters: true
```
