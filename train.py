import argparse
import json
import torch
import torchvision
import math
import cv2
import os
import re
import importlib
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import plotly.express as px

from torch.utils.data import DataLoader
from dataset import HistImage
from tqdm import tqdm
from scipy.stats import pearsonr as corr
from sklearn.metrics import r2_score, mean_squared_error
from utils import Average, pdist, cos

from torchmetrics.regression import LogCoshError
from torchmetrics.regression import MeanSquaredLogError
from torchmetrics.regression import MeanSquaredError

# Argument parser for input arguments like file path, GPU, epochs, and model name
parser = argparse.ArgumentParser()
parser.add_argument('--inp_file', type=str)  # Configuration file
parser.add_argument('--gpu', type=int, default=0)  # GPU device number
parser.add_argument('--fold', type=int, default=0)  # Fold number for cross-validation
parser.add_argument('--epochs', type=int, default=50)  # Number of training epochs
parser.add_argument('--model', type=str, default='my_model')  # Model name

args = parser.parse_args()

# Dynamically import the specified model
custom_module = importlib.import_module(args.model)
Base_AE = custom_module.Base_AE
CustomModel = custom_module.CustomModel
MODEL_NAME = custom_module.MODEL_NAME

# Load configurations from the input file
with open(args.inp_file, 'r') as f:
    configs = json.load(f)

# Meta variables 
weight_path = 'weight_of_final_autoencoder_model.pth'
#from the configuration file
fold = args.fold
n_epochs = args.epochs
exp_path = configs['exp_path']  # Path to experession data
img_path = configs['img_path']  # Path to image tiles
lr_rate = configs['lr_rate']  # Learning rate
gamma_rate = configs['gamma_rate']  # Gamma rate for learning rate scheduler
load_batch = configs['load_batch']  # Batch size for data loading
n_worker = configs['n_worker']  # Number of workers for data loading
inp_dir = configs['inp_dirs']  # Input directory
out_dir = configs['out_dirs']  # Output directory
model_name = MODEL_NAME  # Model name from imported module
config_ = configs["config_file"]  # Configuration file identifier
loss_fun = configs['loss_fun']  # Loss function type

# Set the device to GPU if available
device = torch.device(f"cuda:{args.gpu}")

# Paths to save training and validation results
train_res_path = f'{out_dir}train_{config_}--{model_name}_stnet128_fold_{fold}_epoch_{n_epochs}.csv'
valid_res_path = f'{out_dir}valid_{config_}--{model_name}_stnet128_fold_{fold}_epoch_{n_epochs}.csv'

# Load experimental data and image paths, and split data into training and validation sets
exp_df = pd.read_csv(exp_path, index_col=0)
temp_df = pd.read_csv(img_path, index_col=0)
train_df = temp_df[temp_df.fold != fold]
valid_df = temp_df[temp_df.fold == fold]

# Print the size of the training and validation datasets
train_df.shape, valid_df.shape

# Data transformations (e.g., normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load training and validation data using the custom dataset class
train_data = HistImage(train_df, exp_df, True, transform=transform)
valid_data = HistImage(valid_df, exp_df, True, transform=transform)
print(len(train_data), len(valid_data))

# Create DataLoaders for training and validation datasets
train_loader = DataLoader(dataset=train_data, batch_size=load_batch, shuffle=True, num_workers=n_worker)
valid_loader = DataLoader(dataset=valid_data, batch_size=load_batch, shuffle=True, num_workers=n_worker)

# Load pre-trained model weights and freeze encoder layers
no_of_output = exp_df.shape[1]

base_model = Base_AE()  # Base autoencoder model
base_model.load_state_dict(torch.load(weight_path))  # Load weights into the model
model = CustomModel(base_model.encoder, num_classes=no_of_output).to(device)  # Custom model using encoder
for param in model.encoder.parameters():
    param.requires_grad = False  # Freeze encoder parameters

# Optimizer setup
optimizer = optim.Adam(model.parameters(), lr=lr_rate)

# Choose the loss function based on the configuration
if loss_fun == 'mse':
    criterion = nn.MSELoss()
elif loss_fun == 'rmse':
    criterion = MeanSquaredError(squared=False).to(device)
elif loss_fun == 'dist':
    criterion = nn.PairwiseDistance(p=3)
elif loss_fun == 'huber':
    criterion = nn.HuberLoss(reduction='mean', delta=1.0)
elif loss_fun == 'logcosh':
    criterion = LogCoshError()
elif loss_fun == 'dist':
    criterion = nn.PairwiseDistance(p=4)

# Training loop
Train_Results = []  # List to store training results
Valid_Results = []  # List to store validation results
valid_loss_min = np.Inf  # Tracker for the minimum validation loss

# Move the model to the selected device (GPU)
model.to(device)

for epoch in range(1, n_epochs + 1):
    # Initialize loss variables for training and validation
    train_loss = 0.0
    valid_loss = 0.0

    # Training phase
    model.train()
    train_pred = torch.empty((0))  # Placeholder for training predictions
    train_act = torch.empty((0))  # Placeholder for training targets
    valid_pred = torch.empty((0))  # Placeholder for validation predictions
    valid_act = torch.empty((0))  # Placeholder for validation targets

    train_iterator = tqdm(train_loader)
    for batch_id, data in enumerate(train_iterator):
        # Move data to GPU
        image, target = data['image'].to(device), data['target'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(image)

        # Compute loss and backpropagate
        loss = criterion(output, target)
        loss.backward()

        # Update weights
        optimizer.step()

        # Compute average training loss
        train_loss = train_loss + ((1 / (batch_id + 1)) * (loss.data - train_loss))

        # Append predictions and actual values
        train_pred = torch.cat((train_pred, output.detach().cpu()), dim=0)
        train_act = torch.cat((train_act, target.detach().cpu()), dim=0)

        # Update progress bar
        train_iterator.set_postfix(epoch=epoch, current_loss=loss.item(), total_loss=train_loss.detach().item())

    # Calculate metrics for training data
    mse_error = mean_squared_error(train_act, train_pred)
    dist_error = pdist(train_act, train_pred).mean().item()
    cos_error = cos(train_act, train_pred).mean().item()
    rsquare_val = r2_score(train_act, train_pred)
    corr_val = Average([corr(train_act[i], train_pred[i])[0] for i in range(len(train_act))])

    # Store training results
    train_entry = {'Epoch': epoch, 'Loss': train_loss.item(), 'MSE': mse_error, 'Distance': dist_error,
                   'Cosine': cos_error, 'Rsquare': rsquare_val, 'Correlation': corr_val}
    Train_Results.append(train_entry)

    # Validation phase
    model.eval()
    for batch_id, data in enumerate(valid_loader):
        image, target = data['image'].to(device), data['target'].to(device)
        output = model(image)
        loss = criterion(output, target)
        valid_loss = valid_loss + ((1 / (batch_id + 1)) * (loss.data - valid_loss))

        valid_pred = torch.cat((valid_pred, output.detach().cpu()), dim=0)
        valid_act = torch.cat((valid_act, target.detach().cpu()), dim=0)

    # Calculate metrics for validation data
    mse_error = mean_squared_error(valid_act, valid_pred)
    dist_error = pdist(valid_act, valid_pred).mean().item()
    cos_error = cos(valid_act, valid_pred).mean().item()
    rsquare_val = r2_score(valid_act, valid_pred)
    corr_val = Average([corr(valid_act[i], valid_pred[i])[0] for i in range(len(valid_act))])

    # Store validation results
    valid_entry = {'Epoch': epoch, 'Loss': valid_loss.item(), 'MSE': mse_error, 'Distance': dist_error,
                   'Cosine': cos_error, 'Rsquare': rsquare_val, 'Correlation': corr_val}
    Valid_Results.append(valid_entry)

    # Save the model if validation loss decreases
    val_loss = np.round(valid_loss.item(), 4)
    if val_loss <= valid_loss_min:
        all_files = os.listdir(out_dir + "weights")
        my_pattern = f"train_{config_}--{model_name}_stnet128_fold_{fold}_epoch_{n_epochs}_score_\\d*\\.?\\d+.pth"
        for f in all_files:
            if re.search(my_pattern, f):
                os.remove(f"{out_dir}weights/{f}")

        weight_path = f'{out_dir}weights/train_{config_}--{model_name}_stnet128_fold_{fold}_epoch_{n_epochs}_score_{val_loss}.pth'
        print(f'Validation loss decreased ({valid_loss_min} --> {val_loss}).  Saving model ...')
        torch.save(model.state_dict(), weight_path)
        valid_loss_min = val_loss

# Save training and validation results to CSV files
pd.DataFrame(Train_Results).to_csv(train_res_path, index=None)
pd.DataFrame(Valid_Results).to_csv(valid_res_path, index=None)
