# from __future__ import absolute_import
import os
import re
import threading
from glob import glob
from natsort import natsorted
from pathlib import Path
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset

from stardist_tools import fill_label_holes
from stardist_tools.csbdeep_utils import normalize

from stardist_tools.sample_patches import sample_patches
from stardist_tools.utils import edt_prob, mask_to_categorical
from stardist_tools.geometry import star_dist3D, star_dist

from .utils import load_img, TimeTracker, seed_worker, get_files
from ..models.transforms import get_params, get_transforms


def get_dataloader(opt, data_root, annotation_df, rays=None, is_train_loader=True, augmenter=None):
    if opt.dataset_class == "StarDistData3D":
        dataset_class = StarDistData3D
    elif opt.dataset_class == "StarDistData4D":
        dataset_class = StarDistData4D
    elif opt.dataset_class == "StarDistData4DChannels":
        dataset_class = StarDistData4DChannels
        
    else:
        raise ValueError(f"Unknown dataset class: {opt.dataset_class}")
    
    dataset = dataset_class(
        opt=opt,
        data_root=data_root,
        annotation_df=annotation_df,
        rays=rays,
        augmenter=augmenter,
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=is_train_loader
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=True,
        sampler=sampler
    )

    return dataloader


def get_train_val_dataloaders(opt, rays=None, train_augmenter=None):
    data_root = opt.data_root
    train_df = pd.read_csv(opt.train_split, keep_default_na=False)
    val_df = pd.read_csv(opt.val_split, keep_default_na=False)

    try:
        val_opt = deepcopy(opt())
    except:
        val_opt = deepcopy(opt)
    assert id(val_opt) != id(opt)

    for attr in ["preprocess_val", "intensity_factor_range_val", "intensity_bias_range_val", "scale_limit_val",
                 "resize_to_val", "crop_size_val"]:
        if hasattr(val_opt, attr):
            setattr(val_opt, attr.replace("_val", ""), getattr(val_opt, attr))

    train_dataloader = get_dataloader(opt, data_root, train_df, rays, is_train_loader=True, augmenter=train_augmenter)
    val_dataloader = get_dataloader(val_opt, data_root, val_df, rays, is_train_loader=False)

    train_dataloader.dataset.split = "train"
    val_dataloader.dataset.split = "val"

    return train_dataloader, val_dataloader


class StarDistDataBase(Dataset):
    def __init__(
            self,
            opt,
            data_root,
            annotation_df, 
            augmenter=None
    ):
        super().__init__()

        if opt.n_classes is not None:
            raise NotImplementedError('Multiclass training not implemented yet')

        self.opt = opt
        self.image_ndim = opt.image_ndim

        self.n_channel = opt.n_channel
        self.sd_mode = 'cpp'

        self.grid = tuple(opt.grid[-3:])
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in opt.grid[-3:])
        self.anisotropy = opt.anisotropy

        self.data_root = data_root
        self.annotation_df = annotation_df

        if augmenter is None:
            augmenter = lambda *args: args
        assert callable(augmenter), "augmenter must be None or callable."
        self.augmenter = augmenter
    
        self.time_tracker = TimeTracker()

    def __len__(self):
        return len(self.annotation_df)

    def get_image(self, image_path, normalize_channel="independently"):
        """
        Load and process image at given index.
        
        Args:
            image_path: Path to the image file
            normalize_channel: How to normalize channels ("independently", "jointly", "none")
            
        Returns:
            image: Processed image array with shape (n_channel, *spatial_dims)
        """
        assert normalize_channel in ("independently", "jointly", "none"), normalize_channel

        image = np.load(image_path).astype(np.float32)

        if normalize_channel != "none":
            axis_norm = (0, 1, 2)
            image = normalize(image, 1, 99.8, axis=axis_norm, clip=True)

        patch_size = self.opt.patch_size
        if patch_size is None:
            # Use first 3 dimensions as spatial dimensions for 3D
            patch_size = image.shape[:self.image_ndim]

        ndim = len(patch_size)
        assert ndim == 3, f"len(patch_size={patch_size}) is not 3)"
        assert image.ndim in (ndim, ndim + 1), f"image.ndim not in ({(ndim, ndim + 1)}). image.shape={image.shape}"

        if image.ndim == ndim:
            n_channel = 1
            image = image[np.newaxis]
        else:
            channel_dim = -(self.image_ndim + 1)
            n_channel = image.shape[channel_dim]

        return image

    def get_mask(self, mask_path):
        """
        Load and process mask at given index.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            mask: Processed mask array with spatial dimensions
        """
        mask = np.load(mask_path).astype(np.uint16)

        patch_size = self.opt.patch_size
        if patch_size is None:
            patch_size = mask.shape[:self.image_ndim]

        ndim = len(patch_size)
        assert ndim == 3, f"len(patch_size={patch_size}) is not 3)"
        assert mask.ndim == ndim, f"mask.ndim != {ndim}. mask.shape={mask.shape}"

        return mask

    def channels_as_tuple(self, x):
        return tuple(x[i] for i in range(x.shape[0]))

    def transform(self, image, mask):
        patch_size = list(mask.shape)[-3:]

        transform_param = get_params(self.opt, patch_size)
        transform_image = get_transforms(self.opt, transform_param, is_mask=False)
        transform_mask = get_transforms(self.opt, transform_param, is_mask=True)

        image = transform_image(image)
        mask = transform_mask(mask)

        return image, mask

    def load_and_sample_data(self, idx):
        """Load and sample data. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_and_sample_data")

    def get_anisotropy_for_edt(self):
        """Get anisotropy for EDT computation. Can be overridden by subclasses."""
        return self.anisotropy

    def format_final_image(self, image):
        """Format the final image output. Can be overridden by subclasses."""
        return image

    def get_image_mask_paths(self, idx):
        """Get image and mask paths for output. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_image_mask_paths")

    def __getitem__(self, idx):
        self.time_tracker.tic("loading")
        image, mask, patch_size, n_channel = self.load_and_sample_data(idx)
        mask = mask.astype(np.uint16)
        ndim = len(patch_size)
        self.time_tracker.tac("loading")

        assert image.shape[-ndim:] == tuple(patch_size), (image.shape, patch_size)
        assert mask.shape[-ndim:] == tuple(patch_size), (mask.shape, patch_size)

        self.time_tracker.tic("augmenting")
        image, mask = self.augmenter(image, mask)
        image, mask = self.transform(image, mask)
        self.time_tracker.tac("augmenting")

        self.time_tracker.tic("edt_computing")
        prob = edt_prob(mask, anisotropy=self.get_anisotropy_for_edt())[self.ss_grid[1:]]
        self.time_tracker.tac("edt_computing")

        self.time_tracker.tic("distance_computing")
        dist = star_dist3D(mask, self.rays, mode=self.sd_mode, grid=self.grid)
        dist = np.moveaxis(dist, -1, 0)
        self.time_tracker.tac("distance_computing")

        prob_class = None
        if self.opt.n_classes is not None:
            raise NotImplementedError('Multiclass support not implemented yet')

        image_path, mask_path = self.get_image_mask_paths(idx)

        image = np.clip(image, 1e-8, 1)
        
        item = dict()
        item['image_path'] = image_path.stem if isinstance(image_path, Path) else image_path
        item['mask_path'] = mask_path.stem if isinstance(mask_path, Path) else mask_path
        item['image'] = self.format_final_image(image).astype("float32")
        item['mask'] = mask.astype("float32")

        item["prob"] = prob[np.newaxis]
        item["dist"] = dist
        if prob_class is not None:
            item["prob_class"] = prob_class

        return item


class StarDistData3D(StarDistDataBase):
    def __init__(
            self,
            opt,
            data_root,
            annotation_df,
            rays=None,
            augmenter=None
    ):
        super().__init__(opt=opt, data_root=data_root, annotation_df=annotation_df, augmenter=augmenter)
        if rays is None:
            rays = star_dist.get_default_rays(self.opt.image_ndim)
        self.rays = rays

        if self.opt.n_classes is not None:
            raise NotImplementedError('Multiclass support not implemented yet')

    def load_and_sample_data(self, idx):
        """Load and sample data for 3D case."""
        image_path = os.path.join(self.data_root, self.annotation_df.iloc[idx]['image'])
        mask_path = os.path.join(self.data_root, self.annotation_df.iloc[idx]['mask'])

        image, mask = self.get_image(image_path), self.get_mask(mask_path)

        channel_dim = - (self.image_ndim + 1)
        n_channel = image.shape[channel_dim]

        patch_size = self.opt.patch_size
        if patch_size is None:
            patch_size = mask.shape

        ndim = len(patch_size)

        self.time_tracker.tic("sampling")
        mask, *image_channels = sample_patches(
            (mask,) + self.channels_as_tuple(image),
            patch_size=patch_size,
            n_samples=1,
        )
        mask = mask[0]

        image = np.concatenate(image_channels, axis=0)

        assert image.shape[0] == n_channel, image.shape
        self.time_tracker.tac("sampling")

        return image, mask, patch_size, n_channel

    def get_image_mask_paths(self, idx):
        """Get image and mask paths for 3D case."""
        image_path = os.path.join(self.data_root, self.annotation_df.iloc[idx]['image'])
        mask_path = os.path.join(self.data_root, self.annotation_df.iloc[idx]['mask'])
        return image_path, mask_path


class StarDistData4D(StarDistDataBase):
    def __init__(
            self,
            opt,
            data_root,
            annotation_df,
            rays,
            augmenter=None
    ):
        super().__init__(opt=opt, data_root=data_root, annotation_df=annotation_df, augmenter=augmenter)
        assert rays is not None
        self.rays = rays

    def load_and_sample_data(self, idx):
        """Load and sample data for 4D case."""
        image_path_target = os.path.join(self.data_root, self.annotation_df.iloc[idx]['image_target'])
        mask_path_target = os.path.join(self.data_root, self.annotation_df.iloc[idx]['mask_target'])
        image_target = self.get_image(image_path_target)
        mask_target = self.get_mask(mask_path_target)

        image_path_before = self.annotation_df.iloc[idx]['image_before']
        if image_path_before:
            has_before = True
            image_path_before = os.path.join(self.data_root, image_path_before)
            image_before = self.get_image(image_path_before)
        else:
            has_before = False

        image_path_after = self.annotation_df.iloc[idx]['image_after']
        if image_path_after:
            has_after = True
            image_path_after = os.path.join(self.data_root, image_path_after)
            image_after = self.get_image(image_path_after)
        else:
            has_after = False

        if np.random.uniform() < 0.1:
            has_before = False
        if np.random.uniform() < 0.1:
            has_after = False

        channel_dim = - (self.image_ndim + 1)
        n_channel = image_target.shape[channel_dim]

        patch_size = self.opt.patch_size
        if patch_size is None:
            patch_size = mask_target.shape

        ndim = len(patch_size)

        self.time_tracker.tic("sampling")
        # Stack before, target, and after images and masks as channels for sampling and augmentation
        images_to_sample = self.channels_as_tuple(image_target)
        masks_to_sample = (mask_target,)

        if has_before:
            images_to_sample = self.channels_as_tuple(image_before) + images_to_sample
        
        if has_after:
            images_to_sample = images_to_sample + self.channels_as_tuple(image_after)

        stacked_image_masks = sample_patches(
            masks_to_sample + images_to_sample,
            patch_size=patch_size,
            n_samples=1,
        )
        
        # stacked_image_masks is a tuple
        mask_stacked = np.concatenate(stacked_image_masks[:len(masks_to_sample)], axis=0)
        image_stacked = np.concatenate(stacked_image_masks[len(masks_to_sample):], axis=0)

        # Unstack the sampled image and mask back to separate before, target, and after
        mask_target = mask_stacked[0]

        current_image_idx = 0
        if has_before:
            image_before = image_stacked[current_image_idx:current_image_idx + n_channel]
            current_image_idx += n_channel
        
        image_target = image_stacked[current_image_idx:current_image_idx + n_channel]
        current_image_idx += n_channel

        if has_after:
            image_after = image_stacked[current_image_idx:current_image_idx + n_channel]

        # Create zero arrays if needed, after all transformations
        if not has_before:
            image_before = np.zeros_like(image_target)
        
        if not has_after:
            image_after = np.zeros_like(image_target)

        # Ensure the sampled image and mask have the correct shapes
        assert image_target.shape[0] == n_channel, image_target.shape
        assert image_before.shape[0] == n_channel, image_before.shape
        assert image_after.shape[0] == n_channel, image_after.shape
        assert image_target.shape[-ndim:] == tuple(patch_size), (image_target.shape, patch_size)
        assert image_before.shape[-ndim:] == tuple(patch_size), (image_before.shape, patch_size)
        assert image_after.shape[-ndim:] == tuple(patch_size), (image_after.shape, patch_size)
        assert mask_target.shape[-ndim:] == tuple(patch_size), (mask_target.shape, patch_size)
        self.time_tracker.tac("sampling")

        image = np.stack((image_before, image_target, image_after), axis=1)

        return image, mask_target, patch_size, n_channel

    def get_image_mask_paths(self, idx):
        """Get image and mask paths for 4D case."""
        image_path_target = os.path.join(self.data_root, self.annotation_df.iloc[idx]['image_target'])
        mask_path_target = os.path.join(self.data_root, self.annotation_df.iloc[idx]['mask_target'])
        return image_path_target, mask_path_target


class StarDistData4DChannels(StarDistDataBase):
    def __init__(
            self,
            opt,
            data_root,
            annotation_df,
            rays,
            augmenter=None
    ):
        super().__init__(opt=opt, data_root=data_root, annotation_df=annotation_df, augmenter=augmenter)
        assert rays is not None
        self.rays = rays

    def load_and_sample_data(self, idx):
        """Load and sample data for 4D case."""
        image_path_target = os.path.join(self.data_root, self.annotation_df.iloc[idx]['image_target'])
        mask_path_target = os.path.join(self.data_root, self.annotation_df.iloc[idx]['mask_target'])
        image_target = self.get_image(image_path_target)
        mask_target = self.get_mask(mask_path_target)

        image_path_before = self.annotation_df.iloc[idx]['image_before']
        if image_path_before:
            has_before = True
            image_path_before = os.path.join(self.data_root, image_path_before)
            image_before = self.get_image(image_path_before)
        else:
            has_before = False

        image_path_after = self.annotation_df.iloc[idx]['image_after']
        if image_path_after:
            has_after = True
            image_path_after = os.path.join(self.data_root, image_path_after)
            image_after = self.get_image(image_path_after)
        else:
            has_after = False

        if np.random.uniform() < 0.1:
            has_before = False
        if np.random.uniform() < 0.1:
            has_after = False

        channel_dim = - (self.image_ndim + 1)
        n_channel = image_target.shape[channel_dim]

        patch_size = self.opt.patch_size
        if patch_size is None:
            patch_size = mask_target.shape

        ndim = len(patch_size)

        self.time_tracker.tic("sampling")
        # Stack before, target, and after images and masks as channels for sampling and augmentation
        images_to_sample = self.channels_as_tuple(image_target)
        masks_to_sample = (mask_target,)

        if has_before:
            images_to_sample = self.channels_as_tuple(image_before) + images_to_sample
        
        if has_after:
            images_to_sample = images_to_sample + self.channels_as_tuple(image_after)

        stacked_image_masks = sample_patches(
            masks_to_sample + images_to_sample,
            patch_size=patch_size,
            n_samples=1,
        )
        
        # stacked_image_masks is a tuple
        mask_stacked = np.concatenate(stacked_image_masks[:len(masks_to_sample)], axis=0)
        image_stacked = np.concatenate(stacked_image_masks[len(masks_to_sample):], axis=0)

        # Unstack the sampled image and mask back to separate before, target, and after
        mask_target = mask_stacked[0]

        current_image_idx = 0
        if has_before:
            image_before = image_stacked[current_image_idx:current_image_idx + n_channel]
            current_image_idx += n_channel
        
        image_target = image_stacked[current_image_idx:current_image_idx + n_channel]
        current_image_idx += n_channel

        if has_after:
            image_after = image_stacked[current_image_idx:current_image_idx + n_channel]

        # Create zero arrays if needed, after all transformations
        if not has_before:
            image_before = np.zeros_like(image_target)
        
        if not has_after:
            image_after = np.zeros_like(image_target)

        # Ensure the sampled image and mask have the correct shapes
        assert image_target.shape[0] == n_channel, image_target.shape
        assert image_before.shape[0] == n_channel, image_before.shape
        assert image_after.shape[0] == n_channel, image_after.shape
        assert image_target.shape[-ndim:] == tuple(patch_size), (image_target.shape, patch_size)
        assert image_before.shape[-ndim:] == tuple(patch_size), (image_before.shape, patch_size)
        assert image_after.shape[-ndim:] == tuple(patch_size), (image_after.shape, patch_size)
        assert mask_target.shape[-ndim:] == tuple(patch_size), (mask_target.shape, patch_size)
        self.time_tracker.tac("sampling")

        image = np.concatenate((image_before, image_target, image_after), axis=0)

        return image, mask_target, patch_size, n_channel

    def get_image_mask_paths(self, idx):
        """Get image and mask paths for 4D case."""
        image_path_target = os.path.join(self.data_root, self.annotation_df.iloc[idx]['image_target'])
        mask_path_target = os.path.join(self.data_root, self.annotation_df.iloc[idx]['mask_target'])
        return image_path_target, mask_path_target