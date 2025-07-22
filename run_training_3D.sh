#!/bin/bash

module purge
module load modules/2.3
module load python 
module load cuda cudnn nccl

source ~/envs/stardist/bin/activate

# Slurm command: sbatch -p gpu --nodes=1 --gpus=4 -C a100-80gb --cpus-per-gpu=8 run_training_3D.sh

python -u `which torchrun` \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 4 \
    train.py --yaml_conf ./confs/train_convnext_unet_base-3D.yaml
