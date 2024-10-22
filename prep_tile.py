#!/usr/bin/env python
# coding: utf-8

# Define parameters for the process
data_type = 'train'  # The type of data (train/test)
tile_size = 128  # Size of each image tile (128x128 pixels)

# Paths for input raw images, output tiled images, and template image for normalization
RAW_TILE = f'Replace your raw data path/{data_type}/'
TILE_PATH = f'Provide path for storing tiled output images/{data_type}/'
TEMPLATE_IMG = 'image_normalization_template_slide_path'
NORM_METHOD = 'vahadane'  # Method for stain normalization

# Function to scale individual RGB channels
def scale_rgb(img, r_scale, g_scale, b_scale):
    # Split image into individual RGB channels
    source = img.split()
    R, G, B = 0, 1, 2
    
    # Scale the Red, Green, and Blue channels by their respective scale factors
    red = source[R].point(lambda i: i * r_scale)
    green = source[G].point(lambda i: i * g_scale)
    blue = source[B].point(lambda i: i * b_scale)
    
    # Merge the scaled channels back into an RGB image
    return Image.merge('RGB', [red, green, blue])

# Function to remove color cast from the image
def remove_colour_cast(img):
    img = img.convert('RGB')  # Ensure the image is in RGB mode
    img_array = np.array(img)  # Convert image to NumPy array for manipulation
    
    # Calculate 99th percentile pixel values for each RGB channel
    rp = np.percentile(img_array[:, :, 0].ravel(), q=99)
    gp = np.percentile(img_array[:, :, 1].ravel(), q=99)
    bp = np.percentile(img_array[:, :, 2].ravel(), q=99)
    
    # Scale the image based on the percentile values to remove color cast
    return scale_rgb(img, 255 / rp, 255 / gp, 255 / bp)

# Function to create directories if they do not exist
def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)  # Create the directory if it doesn't exist

# Function to tile an image into smaller sections
def tile(image, out_dir, sample):
    # Get the dimensions of the image
    width, height = image.size
    
    # Loop through the image, creating tiles of size (tile_size x tile_size)
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            
            # Define the boundaries of the current tile
            right = min(x + tile_size, width)
            bottom = min(y + tile_size, height)

            # Crop the tile from the image
            box = (x, y, right, bottom)
            tile = image.crop(box)

            # Ensure the tile size is correct (skip if the tile is smaller than expected)
            if tile.size == (tile_size, tile_size):
                tile_name = f'{sample}--tile_{x}x{y}'
                
                # Save the tile to the output directory in JPEG format
                tile.save(os.path.join(out_dir, tile_name + '.jpeg'), 'JPEG')

# Function to apply tiling on an image after normalization
def do_tile(img_path):
    img = Image.open(img_path)  # Open the image
    sample = img_path.split('/')[-1].split('.')[0]  # Extract sample name from file path
    
    # Remove color cast from the image
    img_uncast = remove_colour_cast(img)
    
    # Standardize luminosity and apply stain normalization
    img_std = LuminosityStandardizer.standardize(np.array(img_uncast))
    transformed = normalizer.transform(img_std)
    img = Image.fromarray(transformed)  # Convert back to image format
    
    # Create the output directory for the sample's tiles
    tile_out = os.path.join(TILE_PATH, sample)
    mkdirs(tile_out)
    
    # Tile the image and save the tiles
    tile(img, tile_out, sample)

# Import necessary libraries and modules
from PIL import Image
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from staintools import StainNormalizer, LuminosityStandardizer

import os
import openslide
import numpy as np
import matplotlib.pyplot as plt

# Set maximum number of pixels to avoid large image warnings
Image.MAX_IMAGE_PIXELS = 933120000

# Get list of raw image files to be tiled
my_files = glob(RAW_TILE + '*/*.png')
len(my_files)  # Print number of files found

# Load the template image for stain normalization
template = Image.open(TEMPLATE_IMG)

# Initialize stain normalizer with the specified method
normalizer = StainNormalizer(method=NORM_METHOD)

# Standardize the template image for normalization
template_std = LuminosityStandardizer.standardize(np.array(template))
normalizer.fit(template_std)

# Apply tiling in parallel on all files using the do_tile function
Parallel(n_jobs=10, verbose=1)(delayed(do_tile)(file) for file in my_files)
