# PyTorch StarDist
This repository contains PyTorch implementations for StarDist with 3D and 4D (3D + time) input images. The backbone has been updated to a more modern ConvNeXT architecture.

- Heavily inspired by [pytorch-stardist](https://github.com/hthierno/pytorch-stardist/tree/main)

- Original StarDist3D paper 
[*Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy*](http://openaccess.thecvf.com/content_WACV_2020/papers/Weigert_Star-convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.pdf).  


## Installation

You should have a C++ compiler installed as this code relies on C/C++ extensions that need to be compiled.

Follow this step to install pytorch stardist:

1. Prereqs: Python 3.10.13, C++ compiler, your choice of environment manager
2. Download the repo
3. Activate the environment and run `pip install -r requirements.txt`
4. `cd stardist_tools` and run `python setup.py install`


## Pretrained Weights
We release weights for convnext_unet_base-3D trained on [BlastoSPIM1](https://blastospim.flatironinstitute.org/html/) for 500 epochs [here](https://drive.google.com/file/d/1vym7YDghXiCip_9pQvlvvCO1sk9NiO8d/view?usp=sharing).