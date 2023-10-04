import torch
import torch as th
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import blobfile as bf

# Source paths
fold = 3
src_images = f"/scratch/as3ek/datasets/pannuke/npy_files/images/fold{fold}/images.npy"
src_masks = f"/scratch/as3ek/datasets/pannuke/npy_files/masks/fold{fold}/masks.npy"

# Target paths
images_path = "/scratch/as3ek/datasets/pannuke/images/"
classes_path = "/scratch/as3ek/datasets/pannuke/classes/"
instances_path = "/scratch/as3ek/datasets/pannuke/instances/"

# Create splits
test_frac = 0.01
splits = ["train", "test"]
for splt in splits:
    os.makedirs(os.path.join(images_path, splt), exist_ok=True)
    os.makedirs(os.path.join(classes_path, splt), exist_ok=True)
    os.makedirs(os.path.join(instances_path, splt), exist_ok=True)

# Load numpy arrays
images = np.load(src_images)
masks = np.load(src_masks)

# Number of images
num_images = images.shape[0]
num_test_images = int(num_images * test_frac)
num_train_images = num_images - num_test_images

for i in tqdm(range(images.shape[0])):
    # Get PIL image
    image = np.uint8(images[i])
    image_pil = Image.fromarray(image)
    
    # Get instance mask
    instance = np.sum(masks[i][:,:,:-1], axis=2)
    instance_pil = Image.fromarray(np.uint8(instance))
    
    # Get class mask
    mask = masks[i].copy()
    mask[:,:,0][mask[:,:,0] != 0] = 1
    mask[:,:,1][mask[:,:,1] != 0] = 2
    mask[:,:,2][mask[:,:,2] != 0] = 3
    mask[:,:,3][mask[:,:,3] != 0] = 4
    mask[:,:,4][mask[:,:,4] != 0] = 5
    
    clz = np.sum(mask[:,:,:-1], axis=2)
    # Set overlap to background
    clz[clz > 5] = 0
    
    # Get PIL image
    clz_pil = Image.fromarray(np.uint8(clz))
    
    # Get split
    if i < num_train_images:
        split = splits[0]
    else:
        split = splits[1]

    # Format filename
    image_name = "image_{0:04d}_{1:09d}.png".format(fold, i)
    instance_name = "instance_{0:04d}_{1:09d}.png".format(fold, i)
    class_name = "class_{0:04d}_{1:09d}.png".format(fold, i)
    
    # Save files
    image_pil.save(os.path.join(images_path, split, image_name))
    instance_pil.save(os.path.join(instances_path, split, instance_name))
    clz_pil.save(os.path.join(classes_path, split, class_name))

print("Done!")
