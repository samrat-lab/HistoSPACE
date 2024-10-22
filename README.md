# Project Overview

This project involves processing and analyzing histopathological images and gene expression data, utilizing various machine learning techniques such as autoencoders and custom models. The input data includes the ICIAR 2018 dataset and spatial transcriptomics data, as mentioned in the ST-NET paper.

## Required Libraries
To ensure the environment is set up correctly, install the following libraries:

## Installation using pip:
```
bash

pip install torch torchvision scprep scikit-learn pandas numpy tqdm matplotlib plotly opencv-python
List of Required Libraries:
```
torch – PyTorch for deep learning models.
torchvision – For pre-trained models and image transformations.
scprep – For single-cell RNA-seq preprocessing.
scikit-learn – Machine learning tools for data preprocessing and evaluation.
pandas – Data manipulation and analysis.
numpy – Scientific computing with arrays.
tqdm – Progress bars for loops.
matplotlib & plotly – Visualization libraries.
opencv-python – For image processing.

## Data Requirements
The project requires two main datasets:

- ICIAR 2018 Breast Cancer Dataset
Download from: ICIAR 2018 Dataset
- ST-NET Spatial Transcriptomics Dataset (from the ST-NET paper)
Download from: ST-NET Dataset or from relevant repositories.
After downloading the datasets, configure the paths in the Python scripts accordingly.

## File Descriptions and Execution
1. prep_tile.py
This script processes histological tiles and prepares them for model training. Ensure the correct paths are set for the input and output directories.

Execution:
```
bash

python prep_tile.py --input_dir <path_to_images> --output_dir <path_to_tiles>
--input_dir: Path to the input images.
--output_dir: Path where processed tiles will be saved.
```
2. autoencoder.py
This file contains the implementation of the autoencoder model used for image feature extraction. You can train the autoencoder using this file.

Execution:
```
bash

python autoencoder.py --input_dir <path_to_tiles> --output_dir <path_to_save_model>
```
--input_dir: Directory of the pre-processed tiles.
--output_dir: Directory to save the trained autoencoder model.

3. prepare_expression.py
This script prepares and normalizes gene expression data from the spatial transcriptomics dataset. It also adds labels and performs various pre-processing steps.

Execution:
```
bash

python prepare_expression.py --meta_file <path_to_meta> --count_file <path_to_counts> --output_file <path_to_save_prepared_data>
```
--meta_file: Path to the metadata file (ST-NET).
--count_file: Path to the count matrix (ST-NET).
--output_file: Path to save the pre-processed gene expression data.

4. dataset.py
This file implements a custom PyTorch Dataset for loading the image and gene expression data, using the tiles and expression matrices. You do not need to run this file directly, as it will be imported when training models.

5. my_model.py
This file contains the custom model architecture, which includes the autoencoder and classification layers. It will be used during training.

6. train.py
This script trains the main model using pre-processed image tiles and gene expression data. Ensure the paths to the autoencoder weights, training, and validation datasets are set properly.

Execution:

```
bash

python train.py --inp_file <config_file> --gpu 0 --fold 0 --epochs 50 --model <model_name>

```
--inp_file: Path to the configuration file (JSON format).
--gpu: The GPU ID to use (default is 0).
--fold: The fold number for cross-validation (default is 0).
--epochs: Number of epochs to train.
--model: The name of the model file containing the custom model class.
Configuration File (JSON)
Here is an example structure for the configuration file used in train.py:

```
json

{
  "exp_path": "/path/to/expression/data.csv",
  "img_path": "/path/to/image/data.csv",
  "lr_rate": 0.0001,
  "gamma_rate": 0.1,
  "load_batch": 32,
  "n_worker": 4,
  "inp_dirs": "/path/to/input/",
  "out_dirs": "/path/to/output/",
  "config_file": "config_name",
  "loss_fun": "mse"
}
```
Ensure to replace the paths accordingly in this configuration file based on where your datasets are stored.

Important Data Path Initialization
Make sure that the following variables in your scripts are updated according to your data locations:

RAW_PATH – Path where the raw data is stored.
META_PATH – Path to the metadata file.
CM_PATH – Path to count matrix data.
SPOT_PATH – Path to spot coordinates data.
COORD_PATH – Path to tumor annotation data.
TILE_PATH – Path where tiles will be saved.
fold_df_path – Path to save the final dataset folds.
You can initialize these variables at the top of each file or in a configuration file to ensure consistency across the scripts.

