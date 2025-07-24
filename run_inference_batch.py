import os
import yaml
from pathlib import Path

import torch
import numpy as np

import argparse
import tifffile

# Pytorch StarDist3D
from pytorch_stardist.data.utils import normalize
from pytorch_stardist.models.config import Config3D
from pytorch_stardist.models.stardist3d import StarDist3D
from utils import seed_all, prepare_conf

# Need this even otherwise DDP will complain
os.environ["LOCAL_RANK"] = '0'
os.environ["RANK"] = '0'

def load_model(config_file = "./confs/train_convnext_unet_base-3D.yaml", 
               checkpoint_file = "./model_checkpoints/convnext_unet_base-3D.pth",
               prob_threshold = 0.5,
               nms_threshold = 0.3):
    
    # Load model config
    with open(config_file) as yconf:
        opt = yaml.safe_load(yconf)

    Config = Config3D
    StarDist = StarDist3D
    conf = Config(**opt, allow_new_params=True)
    opt = prepare_conf(conf)

    # Model instanciation
    model = StarDist(opt)
    model.net.load_state_dict(torch.load(checkpoint_file))
    # model = model.to(device)
    model.net.to(model.device)

    # Probability and NMS threshold for segmentation - play around with these
    model.thresholds['prob'] = prob_threshold
    model.thresholds['nms'] = nms_threshold

    return model

def process_file(model, input_filename, output_path, aniso_factor, 
                 patch_size = [256, 256, 256], context = [64, 64, 64]):
    
    file_extension = os.path.splitext(input_filename)[-1]

    if file_extension != ".tif":
        raise ValueError(f"Input image {input_filename} must be a .tif")
    
    output_filename = os.path.splitext(input_filename)[0] + '.label.tif'
    output_filename = os.path.join(output_path, os.path.basename(output_filename))

    print(f"Starting segmentation for: {input_filename}", flush=True)
    
    image = tifffile.imread(input_filename)

    if image.ndim != 3:
        raise ValueError("This segmentation only supports single channel inputs.")

    # Interpolate image to isotropic resolution
    # NOTE: scipy.ndimage.zoom IS REALLY SLOW. CONSIDER REPLACING WITH torch.nn.functional.interpolate
    # image = zoom(image, (aniso_factor, 1, 1), order = 1) # trilinear

    image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    image = torch.nn.functional.interpolate(image, 
                                            scale_factor=(aniso_factor, 1, 1), 
                                            mode='trilinear')
    image = torch.squeeze(image).numpy()

    # Make image dimensions divisible by n (for UNET down and up):
    n = 32
    image_crop = image[:image.shape[0] - image.shape[0] % n, 
                       :image.shape[1] - image.shape[1] % n, 
                       :image.shape[2] - image.shape[2] % n]
    
    # get the difference in size between the original image and the cropped version
    pad_num = np.array(image.shape) - np.array(image_crop.shape)
    
    # Normalize image
    image_crop = normalize(image_crop, 1, 99.8)
    # add channel dim
    image_crop = np.expand_dims(image_crop, 0) 

    # Predict segmentation mask
    # Needs >20GB VRAM (May spike to >40GB idk, may want to req. 80GB gpu)
    labels, res_dict =  model.predict_instance(image_crop, patch_size=patch_size, context=context) 
    labels = labels.astype(np.uint16)

    # pad z dim with zeros to match the input isotropic size (before cropping)
    labels = np.pad(labels, ((0, pad_num[0]), (0, pad_num[1]), (0, pad_num[2])), mode='constant')

    tifffile.imwrite(output_filename, labels, shape=labels.shape, compression='zlib')

    print(f"Segmentation output written to: {output_filename}", flush=True)

def process_directory(model, input_path, output_path, recursive_flag, 
                      aniso_factor, patch_size = [256, 256, 256], context = [64, 64, 64]):
    
    pattern = "**/*.tif" if recursive_flag else "*.tif"
    folder = Path(input_path)

    for file_path in folder.glob(pattern):
        if file_path.is_file():
            process_file(model, 
                         file_path, 
                         output_path, 
                         aniso_factor, 
                         patch_size, 
                         context)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Segments the histone channel of a z-stack.")
    parser.add_argument("input", help="Input file or directory (file(s) you want to segment).")
    parser.add_argument("output", help="Output file or directory (where the segmentation output goes).")
    parser.add_argument("-r", "--recursive", action="store_true", help="Look through the input directory recursively.")
    parser.add_argument("--config", default="./confs/train_convnext_unet_base-3D.yaml", help="Set the model config file here.")
    parser.add_argument("--checkpoint", default="./model_checkpoints/convnext_unet_base-3D.pth", help="Set the model checkpoint file here.")
    parser.add_argument("--ignore_cuda", action="store_true", help="If CUDA is not avaiable, run anyway.")
    parser.add_argument("--resXY", default=0.208, help="Set the XY resolution (um/pixel).")
    parser.add_argument("--resZ", default=2.0, help="Set the Z resolution (um/pixel).")
    parser.add_argument("--patch_size", default=256, help="Patch size for each network evalutation.")
    parser.add_argument("--context", default=64, help="Context size, i.e patch overlap.")
    parser.add_argument("--prob", default=0.5, help="Probability threshold.")
    parser.add_argument("--nms", default=0.3, help="NMS threshold.")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    checkpoint_file = args.checkpoint
    config_file = args.config
    aniso_factor = float(args.resZ) / float(args.resXY)
    patch_size = [int(args.patch_size),] * 3
    context = [int(args.context),] * 3
    prob_threshold = float(args.prob)
    nms_threshold = float(args.nms)

    # check if cuda is available
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available! This little maneuver's gonna cost us 51 years!", flush=True)
        if not args.ignore_cuda:
            return
        
    model = load_model(config_file, checkpoint_file, prob_threshold, nms_threshold)

    # make output directory:
    print("***************************************", flush=True)
    print(f"Output Dir: {output_path}", flush=True)
    print("***************************************", flush=True)
    os.makedirs(output_path, exist_ok=True)

    # Check if the input path is a directory or file
    if os.path.isdir(input_path):
        print(f"Processing Directory: {input_path}", flush=True)
        print("***************************************", flush=True)
        # If input is a directory, process all files in the directory (recursively or not)
        process_directory(model, input_path, output_path, args.recursive, 
                          aniso_factor, patch_size, context)
    elif os.path.isfile(input_path):
        print(f"Processing file: {input_path}", flush=True)
        print("***************************************", flush=True)
        # If input is a compatible file type, process it
        process_file(model, input_path, output_path, 
                     aniso_factor, patch_size, context)
    else:
        raise ValueError(f"Error: {input_path} is not a valid file or directory.")

if __name__ == '__main__':
    main()