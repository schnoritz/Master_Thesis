import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class DoseDataset(Dataset):
    def __init__(self, root_path, csv_path, patch_size, transforms=None):
        self.root_path = root_path
        self.files = pd.read_excel(csv_path)
        self.transforms = transforms
        self.patch_size = patch_size

    
    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        
        self.training_paths, self.target_path = self.get_paths(idx)

        training_arrays, target_array = self.read_arrays()

        idxs = self.get_idxs(training_arrays)

        training_patches, target_patch = self.extract_patch_from3d(training_arrays, target_array, idxs)
        
        return (training_patches.double(), target_patch.double())


    def extract_patch_from3d(self, training, target, idxs):
        
        training_patches = []
        for array in training:
            training_patches.append(array[idxs[0]:idxs[0]+self.patch_size,
                                          idxs[1]:idxs[1]+self.patch_size,
                                          idxs[2]:idxs[2]+self.patch_size])

        target_patch = target[idxs[0]:idxs[0]+self.patch_size,
                              idxs[1]:idxs[1]+self.patch_size,
                              idxs[2]:idxs[2]+self.patch_size]

        target_patch = torch.unsqueeze(target_patch, 0)

        return torch.stack(training_patches), target_patch
            
        
    def get_idxs(self, data_array):
        
        size = data_array[0].shape

        patch_idxs_boundaries = [[0, size[0]-self.patch_size], \
                                 [0, size[1]-self.patch_size], \
                                 [0, size[2]-self.patch_size]]

        idxs = [torch.randint(patch_idxs_boundaries[0][0], patch_idxs_boundaries[0][1], (1,)),
                torch.randint(patch_idxs_boundaries[1][0], patch_idxs_boundaries[1][1], (1,)),
                torch.randint(patch_idxs_boundaries[2][0], patch_idxs_boundaries[2][1], (1,))]

        return idxs
    
    
    def read_arrays(self):  
        
        arrays = []
        for array in self.training_paths:
            with open(array, 'rb') as fin:
                arrays.append(torch.tensor(np.load(fin)))

        with open(self.target_path, 'rb') as fin:
            target_array = torch.tensor(np.load(fin))

        return arrays, target_array

    def get_paths(self, idx):
        ct_path = self.root_path + self.files.iloc[idx, 2]
        depth_path = self.root_path  + self.files.iloc[idx, 3]
        binary_path = self.root_path + self.files.iloc[idx, 4]
        center_path = self.root_path + self.files.iloc[idx, 5]
        source_path = self.root_path + self.files.iloc[idx, 6]
        target_path = self.root_path + self.files.iloc[idx, 7]

        return [ct_path, depth_path, binary_path, center_path, source_path], target_path