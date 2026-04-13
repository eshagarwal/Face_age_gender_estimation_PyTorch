import torch
import numpy as np
from torch.utils.data import Dataset

def process_pixels(pixel_string):
    # 1. Split the string into a list of individual pixel values
    pixel_list = pixel_string.split()
    
    # 2. Convert to a NumPy array of floating-point numbers
    pixel_array = np.array(pixel_list, dtype=np.float32)

    # print(pixel_array.shape)
    
    # 3. Reshape the flat 2304-length array into a 48x48 image
    image = pixel_array.reshape(48, 48)
    
    return image

class FaceDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # 1. Get the row
        row = self.df.iloc[idx]
        
        # 2. Process the pixels (using our function from earlier)
        img = process_pixels(row['pixels'])
        
        # 3. Normalize and convert to PyTorch tensor
        # (CNNs expect: [Channels, Height, Width])
        img_tensor = torch.tensor(img / 255.0).float().unsqueeze(0)
        
        # 4. Prepare labels
        labels = {
            'age': torch.tensor(row['age']).float(),
            'gender': torch.tensor(row['gender']).long(),
            'ethnicity': torch.tensor(row['ethnicity']).long()
        }
        
        return img_tensor, labels