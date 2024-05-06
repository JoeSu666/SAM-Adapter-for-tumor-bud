# Adapting SAM to histopathology images for tumor bud segmentation in colorectal cancer
SPIE Medical Imaging 2024 [Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12933/129330C/Adapting-SAM-to-histopathology-images-for-tumor-bud-segmentation-in/10.1117/12.3006517.full)

Before running, create `load` folder for saving data and `save` folder for saving logs.

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

Download trained [checkpoint](https://wakehealth-my.sharepoint.com/:u:/r/personal/mgurcan_wakehealth_edu/Documents/cialab/sam-adapter-tumor-bud/model_epoch_epoch_35.pth?csf=1&web=1&e=h51GYI).

## Acknowledgements
The part of the code is derived from Explicit Visual Prompt   <a href='https://nifangbaage.github.io/Explicit-Visual-Prompt/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> by 
Weihuang Liu, [Xi Shen](https://xishen0220.github.io/), [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/), and [Xiaodong Cun](https://vinthony.github.io/) by University of Macau and Tencent AI Lab. \
The part of the code is derived from SAM-Adapter   <a href='https://github.com/tianrun-chen/SAM-Adapter-PyTorch/tree/main'><img src='https://img.shields.io/badge/Project-Page-Green'></a> by 
Tianrun Chen et al. from KOKONI, Moxin Technology (Huzhou) Co., LTD , Zhejiang University, Singapore University of Technology and Design, Huzhou University, Beihang University.
