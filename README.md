## Train
```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 1 train.py --config configs/exp.yaml
```

## resume training
```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 1 train.py --config configs/exp_resume.yaml
```


## Test
1. Generate outputs:
```shell
python testf.py --config configs/exp.yaml --model save/_exp/model_epoch_best.pth --runcode chen512_35ep
```
2. Perform mask refinement with morphology transformation and evaluate the final output: SAMA_hovernet_test.ipynb
