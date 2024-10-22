from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Meta variables

train_PATH = f'set tile location'
val_PATH = f'set tile location of validation data'
result_file = 'Provide a file name and path with csv extension'
learning_rate = 0.00001  # Learning rate for the optimizer
epochs = 150  # Number of training epochs
device = torch.device('cuda')  # Use GPU for training if available

# Dataset class definition
class Hist_patch(object):
    # Initialize the class with an image list and optional transformations
    def __init__(self, img_list, transform=None):
        super(Hist_patch, self).__init__()
        self.img_list = img_list  # List of image file paths
        self.sample_size = len(img_list)  # Total number of samples
        self.transform = transform  # Optional image transformations (e.g., normalization)

    # Return the total number of samples
    def __len__(self):
        return self.sample_size

    # Get an individual sample (image) based on the index
    def __getitem__(self, index):
        img_path = self.img_list[index]  # Get the image file path
        img_id = img_path.split('/')[-1].split('.')[0]  # Extract image ID from the path
        img = Image.open(img_path)  # Open the image

        # Apply transformations if specified
        if self.transform is not None:
            img = self.transform(img)

        # Return a dictionary containing the image ID and the transformed image
        return {'id': img_id, 'image': img}

# Model definition (Encoder-Decoder structure)

import torch
import torch.nn as nn

# Encoder definition: performs down-sampling and feature extraction
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # First convolutional layer: 3 input channels, 32 output channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by factor of 2

        # Second convolutional layer: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional layer: 64 input channels, 128 output channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Forward pass through the encoder
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        return x

# Decoder definition: performs up-sampling to reconstruct the image
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # First transpose convolution layer: upsample from 128 to 64 channels
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # Second transpose convolution: upsample from 64 to 32 channels
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # Third transpose convolution: upsample from 32 to 3 channels (RGB output)
        self.conv_transpose3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation to keep pixel values between 0 and 1

    # Forward pass through the decoder
    def forward(self, x):
        x = self.conv_transpose1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv_transpose2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv_transpose3(x)
        x = self.sigmoid(x)
        return x

# Base Autoencoder model combining Encoder and Decoder
class Base_AE(nn.Module):
    def __init__(self):
        super(Base_AE, self).__init__()
        self.encoder = Encoder()  # Encoder instance
        self.decoder = Decoder()  # Decoder instance

    # Forward pass through the autoencoder
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training script

from torchsummary import summary
from dataset import Hist_patch
from glob import glob
from PIL import Image

import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Data preprocessing: normalization and transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5), (0.5))  # Normalize image
])

# Load training and validation data
train_files = glob(train_PATH + '*/*.jpeg')
val_files = glob(val_PATH + '*/*.jpeg')

# Create dataset and dataloaders for training and validation
train_data = Hist_patch(train_files, transform=transform)
val_data = Hist_patch(val_files, transform=transform)

# Define data loaders
train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=10)
valid_loader = DataLoader(dataset=val_data, batch_size=256, shuffle=True, num_workers=10)

# Initialize the autoencoder model
autoencoder = Base_AE().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)  # Adam optimizer

# Training loop
best_val_loss = float('inf')  # Best validation loss initialized to infinity
final_results = []  # Store results for logging

for epoch in range(epochs):
    # Training phase
    train_loss = 0
    autoencoder.train()
    for _id, batch_data in enumerate(train_loader):
        optimizer.zero_grad()  # Zero gradients
        inputs = batch_data['image'].to(device)  # Load inputs
        outputs = autoencoder(inputs)  # Forward pass
        loss = criterion(outputs, inputs)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        train_loss += loss.item()
    train_loss /= len(train_loader)  # Average training loss

    # Validation phase
    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for _id, batch_data in enumerate(valid_loader):
            inputs = batch_data['image'].to(device)  # Load validation inputs
            outputs = autoencoder(inputs)  # Forward pass
            val_loss += criterion(outputs, inputs).item()  # Compute validation loss
    val_loss /= len(valid_loader)  # Average validation loss

    # Print validation loss for the current epoch
    print(f'Epoch [{epoch+1}/{epochs}] - Validation Loss: {val_loss:.4f}')
    final_results.append({'Epoch': epoch, 'train': train_loss, 'val': val_loss})

    # Save model checkpoint if validation loss decreased
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(autoencoder.state_dict(), f'./weight/trained_model_epoch_{epoch}_val_{round(val_loss,4)}.pth')

# Save final results to CSV
pd.DataFrame(final_results).to_csv(result_file, index=None)
print('Training complete.')
