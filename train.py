import gc
import sys
import os
import warnings
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Surpress tensorflow debugging warnings
from tqdm import tqdm
import yaml

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stardist_tools import Rays_GoldenSpiral

from pytorch_stardist.training import train
from pytorch_stardist.data.stardist_dataset import get_train_val_dataloaders
from utils import seed_all, prepare_conf

from pytorch_stardist.models.config import Config3D
from pytorch_stardist.models.stardist3d import StarDist3D

from evaluate import evaluate
import subprocess

def run(args):
    """
    Load configurations from the YAML file specified in the command
    Create the StarDist model
    Load the data
    Perform model training and thresholds optimization
    Perform evaluation on train and validation sets
    """
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    torch.distributed.init_process_group(backend='nccl')
    
    with open(args.yaml_conf) as yconf:
        opt = yaml.safe_load(yconf)

    Config = Config3D
    StarDist = StarDist3D

    conf = Config(**opt, allow_new_params=True)

    # Set random seed
    seed_all(conf.random_seed)

    # process the configuration variables
    opt = prepare_conf(conf)
    
    # Model instanciation
    model = StarDist(opt).to(local_rank)
    model.net = torch.nn.parallel.DistributedDataParallel(model.net, device_ids=[local_rank])
    
    # Loading data
    rays = None
    if model.opt.is_3d:
        rays = Rays_GoldenSpiral(opt.n_rays, anisotropy=opt.anisotropy[-3:])
    
    train_dataloader, val_dataloader = get_train_val_dataloaders(opt, rays)
    
    total_nb_samples = len(train_dataloader.dataset) + (
        len(val_dataloader.dataset) if val_dataloader is not None else 0)
    nb_samples_train = len(train_dataloader.dataset)
    nb_samples_val = total_nb_samples - nb_samples_train
    
    if global_rank == 0:
        print("Total nb samples: ".ljust(40), total_nb_samples)
        print("Train nb samples: ".ljust(40), nb_samples_train)
        print("Val nb samples: ".ljust(40), nb_samples_val)
        print("Train augmentation".ljust(25), ":", train_dataloader.dataset.opt.preprocess)
        print("Val augmentation".ljust(25), ":", val_dataloader.dataset.opt.preprocess)

    # Training
    train(model, train_dataloader, val_dataloader)
    
    if global_rank == 0:
        # Threshold optimization   
        print("Optimizing Thresholds...")
        model.load_state(name='best')

        X, Y = val_dataloader.dataset.get_all_data()
        model.optimize_thresholds(X[:100], Y[:100]) # Very slow so use smaller subset of validation set

        # Evaluation
        log_dir = Path(model.opt.log_dir) / f"{model.opt.name}"

        patch_size = None
        # Uncomment the next line To do inference per patch in order to reduce memory usage. NB: IMPLEMENTED ONLY FOR StarDist3D
        # patch_size=(32, 128, 128) # or patch_size = model.opt.patch_size

        # Evaluation on Validation set
        print("Evaluation on validation set")
        print("\tPredicting...")
        Y_pred = [model.predict_instance(x, patch_size=patch_size)[0] for x in X]
        print("\tEvaluating...")
        stats, fig = evaluate(Y_pred, Y, use_tqdm=False)
        plt.savefig(log_dir / "acc_on_val_set.png")
        plt.close(fig)
        stats = pd.DataFrame(stats)
        stats.to_csv(log_dir / "perf_on_val_set.csv", index=False)
        print(stats)

        print(f"\n\nEvaluation scores saved at <{log_dir.absolute()}>")
    
    torch.distributed.destroy_process_group()

    
def threshold_optimization(args):
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_conf", type=str, help='YAML configuration file.')
    args = parser.parse_args()
    
    run(args)