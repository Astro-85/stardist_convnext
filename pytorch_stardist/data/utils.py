import os
import random
from collections import defaultdict
import time

from pathlib import Path

import numpy as np
import torch
import tifffile
from PIL import Image


class TimeTracker:
    def __init__(self):
        self.dict_start_end = defaultdict(list)
        self.phases = defaultdict(list)

    def tic(self, phase: str):
        self.dict_start_end[phase].append([time.time(), None])

    def tac(self, phase: str):
        end = time.time()
        start = self.dict_start_end[phase][-1][0]
        self.dict_start_end[phase][-1][-1] = end
        self.phases[phase].append(end - start)

    def get_duration(self, phase):
        return self.phases[phase]

    def __getitem__(self, phase):
        return self.phases[phase]


def get_files(
        dirpath,
        extensions=('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF',
                    '.tiff', '.TIFF')
):
    filepaths = []
    assert os.path.isdir(dirpath), '%s is not a valid directory' % dirpath

    for root, _, fnames in sorted(os.walk(dirpath)):
        for fname in fnames:
            if Path(fname).suffix in extensions:
                path = os.path.join(root, fname)
                filepaths.append(Path(path))
    return filepaths


def load_img(path):
    if str(path).lower().endswith("tif") or str(path).lower().endswith("tiff"):
        return load_tif(path)

    im = Image.open(path)
    im = np.array(im)
    if im.ndim == 3 and im.shape[-1] <= 4:
        im = im.swapaxes(-1, 0).swapaxes(-1, 1)

    return im


def save_img(path, im_arr):
    if str(path).lower().endswith("tif") or str(path).lower().endswith("tiff"):
        return save_tif(path, im_arr)

    if im_arr.ndim == 3 and im_arr.shape[0] <= 4:
        im_arr = im_arr.swapaxes(-1, 0).swapaxes(-1, 1)

    Image.fromarray(im_arr).save(path)


def load_tif(path):
    with tifffile.TiffFile(path) as tif:
        return tif.asarray()  # .astype("float32")


def save_tif(path, tif):
    tifffile.imsave(path, tif)


def seed_all(seed):
    """
    Utility function to set seed across all pytorch process for repeatable experiment
    """
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Utility function to set random seed for DataLoader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x


def normalize_minmse(x, target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov = np.cov(x.flatten(),target.flatten())
    alpha = cov[0,1] / (cov[0,0]+1e-10)
    beta = target.mean() - alpha*x.mean()
    return alpha*x + beta