# Project Overview

This is the code repository of HistoSPACE paper, please follow the paper for more details. The input data includes the ICIAR dataset and spatial transcriptomics data, as mentioned in the ST-NET paper.

## Required Libraries
To ensure the environment is set up correctly, install the following libraries:

## Installation using pip:
```
bash

pip install torch torchvision scprep scikit-learn pandas numpy tqdm matplotlib plotly opencv-python

```
List of Required Libraries:

- torch – PyTorch for deep learning models.
- torchvision – For pre-trained models and image transformations.
- scprep – For single-cell RNA-seq preprocessing.
- scikit-learn – Machine learning tools for data preprocessing and evaluation.
- pandas – Data manipulation and analysis.
- numpy – Scientific computing with arrays.
- tqdm – Progress bars for loops.
- matplotlib & plotly – Visualization libraries.
- opencv-python – For image processing.

## Data Requirements
The project requires two main datasets:

- ICIAR Breast Cancer Dataset
- STNet Spatial Transcriptomics Dataset (from the ST-NET paper)

After downloading the datasets, configure the paths in the Python scripts accordingly.

## File Descriptions and Execution

1. prep_tile.py

Run this script after downloading data from ICIAR and STNet and set the data path as `RAW_TILE` for respective data to generate the tiles.

Execution: Edit `RAW_TILE` inside the script and run the following command.
```
bash

python prep_tile.py
```

2. autoencoder.py

This file contains the implementation of the autoencoder model used for image feature extraction. You can train the autoencoder using this file. Use the trained weight file (.pth) in subsquent steps.

Execution: Edit meta variables inside the script before running the command.
```
bash

python build_autoencoder.py
```

3. pre_exp-data.py

This script prepares and normalizes gene expression data from the spatial transcriptomics dataset. It also maps the expression of selected genes with respective stop coordinate (X,Y) in image. Use the generated file as input for model training script.

Execution: Edit meta variables inside the script before running the command.
```
bash

python pre_exp-data.py

```


4. dataset.py

This file implements a custom PyTorch Dataset for loading the image and gene expression data, using the tiles and expression matrices. Need not to run this file directly, as it will be imported when training models.

5. my_model.py

This file contains the custom model architecture, which includes the autoencoder and classification layers. It will be used during training.

6. train.py

This script trains the main model using pre-processed image tiles and gene expression data. Ensure the paths to the autoencoder weights ['weight_path'] is set properly.

Execution: Prepare the configuration file below in .json formate and pass to --inp_file argument

```
config_file.json

{
  "exp_path": "/path/to/expression/data.csv",# file generated from step 3
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

Run the following command to prepare HistoSPACE model in leave one out cross-validation

```
bash

python train.py --inp_file config_file.json --gpu 0 --fold 0 --epochs 50 --model <model_name>

```

Caution: Ensure that all meta variables in the scripts are updated as per the data locations:

# Citation

Please cite our paper:

```
@article{kumar2024histospace,
  title={HistoSPACE: Histology-Inspired Spatial Transcriptome Prediction And Characterization Engine},
  author={Kumar, Shivam and Chatterjee, Samrat},
  journal={arXiv preprint arXiv:2408.03592},
  year={2024}
}

```
