#!/bin/bash

#SBATCH --job-name=infer
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

module purge
module load modules/2.3
module load python
module load cuda cudnn nccl

source /mnt/home/alu10/envs/stardist/bin/activate

# Input and output paths to images and segmentation:
# NOTE: THE OUTPUT DIRECTORY WILL BE CREATED IF IT DOESN'T EXIST
INPUT_PATH="/mnt/ceph/users/ddenberg/MouseProject/MouseData/250621_stack0/ch1long_histone_crop"
OUTPUT_PATH="/mnt/ceph/users/ddenberg/MouseProject/MouseData/250621_stack0/histone_segmentation"

## Looks recursivly through subfolders in the input directory (true/false):
do_recursive=true

## Image resolution in XY and Z:
RES_XY="0.208"
RES_Z="1.0"

# Probability and NMS thresholds for segmentation model:
PROB_THRESHOLD="0.5"
NMS_THRESHOLD="0.3"

# Where the 'run_inference_batch.py' file is located:
SCRIPT_PATH="run_inference_batch.py"

# config and checkpoint paths:
CONFIG_PATH="/mnt/ceph/users/ddenberg/stardist_convnext/confs/train_convnext_unet_base-3D.yaml"
CHECKPOINT_PATH="/mnt/ceph/users/ddenberg/stardist_convnext/model_checkpoints/convnext_unet_base-3D_250724.pth"

if [ "$do_recursive" = true ]; then
    python ${SCRIPT_PATH} ${INPUT_PATH} ${OUTPUT_PATH} \
        -r \
        --config ${CONFIG_PATH} \
        --checkpoint ${CHECKPOINT_PATH} \
        --resXY ${RES_XY} \
        --resZ ${RES_Z} \
        --prob ${PROB_THRESHOLD} \
        --nms ${NMS_THRESHOLD}
else
    python ${SCRIPT_PATH} ${INPUT_PATH} ${OUTPUT_PATH} \
        --config ${CONFIG_PATH} \
        --checkpoint ${CHECKPOINT_PATH} \
        --resXY ${RES_XY} \
        --resZ ${RES_Z} \
        --prob ${PROB_THRESHOLD} \
        --nms ${NMS_THRESHOLD}
fi
