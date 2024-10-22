from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import cv2

class HistImage(Dataset):
    
    # Initialize the dataset class with data paths, labels, training flag, and transformations
    def __init__(self, path_data, exp_data, is_train, transform=None):
        self.path_data = path_data  # DataFrame containing paths to the images
        self.exp_data = exp_data  # DataFrame containing the experimental data (targets)
        self.sample_size = len(self.path_data)  # Total number of samples
        self.is_train = is_train  # Boolean flag indicating if it's a training set
        self.transform = transform  # Optional transformations (e.g., normalization)
    
    # Return the total number of samples in the dataset
    def __len__(self):
        return self.sample_size

    # Get an individual sample based on the index
    def __getitem__(self, index):
        my_index = self.path_data.iloc[index].cm  # Get index corresponding to the sample
        img_path = self.path_data.loc[my_index].img  # Retrieve the image path from DataFrame
        image = Image.open(img_path)  # Open the image using PIL (or cv2 if needed)

        # Apply transformations if provided (e.g., resizing, normalization)
        if self.transform is not None:
            image = self.transform(image)

        # Retrieve the target values (e.g., labels) corresponding to the current sample
        target = self.exp_data.loc[my_index].values
        
        # Convert the target and image data to PyTorch tensors
        target = torch.tensor(target).float()
        case_id = my_index  # Store the case ID for further use
        
        # Return data depending on whether it's a training or testing dataset
        if self.is_train:
            # If training, return image, target, and case ID
            return {"image": image, "target": target, "case_id": case_id}
        else:
            # If not training (i.e., testing or validation), return image and case ID only
            return {"image": image, "case_id": case_id}
