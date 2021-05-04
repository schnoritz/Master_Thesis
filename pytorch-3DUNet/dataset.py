import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
from numpy.random import randint
import matplotlib.pyplot as plt 

class DoseDataset(Dataset):
    def __init__(self, root_path, csv_path, patch_size, transforms=None):
        self.root_path = root_path
        self.files = pd.read_excel(csv_path)
        self.transforms = transforms
        self.patch_size = patch_size
    
    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        
        self.paths = self.get_paths(idx)
        data_arrays = self.read_arrays()

        idxs = self.get_idxs(data_arrays)
        patches = self.extract_patch_from3d(data_arrays, idxs)
        
        return (patches, torch.empty((self.patch_size, self.patch_size, self.patch_size)))
        
    def extract_patch_from3d(self, data, idxs):
        
        patches = []
        for array in data:
            patches.append(array[idxs[0]:idxs[0]+self.patch_size,
                                 idxs[1]:idxs[1]+self.patch_size,
                                 idxs[2]:idxs[2]+self.patch_size])

        return torch.stack(patches)
            
        
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
        for array in self.paths:
            with open(array, 'rb') as fin:
                arrays.append(torch.tensor(np.load(fin)))

        return arrays

    def get_paths(self, idx):
        ct_path = self.root_path + self.files.iloc[idx, 2]
        depth_path = self.root_path  + self.files.iloc[idx, 3]
        binary_path = self.root_path + self.files.iloc[idx, 4]
        center_path = self.root_path + self.files.iloc[idx, 5]
        source_path = self.root_path + self.files.iloc[idx, 6]

        return [ct_path, depth_path, binary_path, center_path, source_path]

        

def test():
    root_dir = "/home/baumgartner/sgutwein84/training_data/training/"
    csv_dir = root_dir + "csv_files.xls"
    my_dataset = DoseDataset(root_dir, csv_dir, 64)
    
    train_set, test_set = torch.utils.data.random_split(
        my_dataset, [int(np.ceil(len(my_dataset)*0.95)), int(np.floor(len(my_dataset)*0.05))])
    train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        print("Generating Next Patch... ")
        plot_patches(data, batch_idx)
        print("Patches Saved!")

def plot_patches(patches, idx):
    
    fig, axs = plt.subplots(1,5)
    img=0
    for b in patches:
        i=0
        img+=1
        for p in b:
            axs[i].imshow(p[:,:,p.shape[2]//2])
            i+=1
            
        plt.savefig("/home/baumgartner/sgutwein84/training_data/output_images/image" + str(img) + str(idx) + ".png")
    
    plt.close()    
    
if __name__ == "__main__":
    test()

