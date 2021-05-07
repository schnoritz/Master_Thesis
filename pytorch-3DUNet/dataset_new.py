import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from time import time

class DoseDataset(Dataset):
    def __init__(self, root_path, csv_path, patch_size, transforms=None):
        self.root_path = root_path
        self.files = pd.read_excel(csv_path, header=None)
        self.transforms = transforms
        self.patch_size = patch_size
        self.all_file_masks = self.load_all_masks()
    
    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        
        file_idx = torch.randint(0,self.all_file_masks.shape[0], (1,))

        idxs = self.get_idxs()

        training_patches, target_patch = self.extract_patch(file_idx, idxs)
        
        return (training_patches.double(), target_patch.double())


    def load_all_masks(self):

        mask_paths, target_path = self.get_paths()
        
        num_files = len(self.files)

        start = time()
        all_masks = []
        for file_idx in range(num_files):
            masks = []
            masks.append(np.load(self.root_path + self.files.iloc[file_idx, 2]))
            masks.append(np.load(self.root_path + self.files.iloc[file_idx, 3]))
            masks.append(np.load(self.root_path + self.files.iloc[file_idx, 4]))
            masks.append(np.load(self.root_path + self.files.iloc[file_idx, 5]))
            masks.append(np.load(self.root_path + self.files.iloc[file_idx, 6]))
            masks.append(np.load(self.root_path + self.files.iloc[file_idx, 7]))
            all_masks.append(masks)

        print(f"It Took {np.round(time()-start)} Seconds to Load the Masks")
        

        return np.array(all_masks)

    def extract_patch(self, file_idx, idxs):
        
        training_patches = []
        for num in range(5):
            training_patches.append(torch.tensor(self.all_file_masks[file_idx][num][idxs[0]:idxs[0]+self.patch_size,
                                                                       idxs[1]:idxs[1]+self.patch_size,
                                                                       idxs[2]:idxs[2]+self.patch_size]))

        target_patch = torch.tensor(self.all_file_masks[file_idx][5][idxs[0]:idxs[0]+self.patch_size,
                                                          idxs[1]:idxs[1]+self.patch_size,
                                                          idxs[2]:idxs[2]+self.patch_size])
        
        target_patch = torch.unsqueeze(target_patch, 0)

        return torch.stack(training_patches), target_patch
            
        
    def get_idxs(self):

        size = self.all_file_masks.shape[2:]

        
        patch_idxs_boundaries = [[0, size[0]-self.patch_size], \
                                 [0, size[1]-self.patch_size], \
                                 [0, size[2]-self.patch_size]]

        idxs = [torch.randint(patch_idxs_boundaries[0][0], patch_idxs_boundaries[0][1], (1,)),
                torch.randint(patch_idxs_boundaries[1][0], patch_idxs_boundaries[1][1], (1,)),
                torch.randint(patch_idxs_boundaries[2][0], patch_idxs_boundaries[2][1], (1,))]

        return idxs

    def get_paths(self):
        ct_path = "/".join((self.root_path + self.files.iloc[0, 2]).split("/")[:-1]) + "/"
        depth_path = "/".join((self.root_path  + self.files.iloc[0, 3]).split("/")[:-1]) + "/"
        binary_path = "/".join((self.root_path + self.files.iloc[0, 4]).split("/")[:-1]) + "/"
        center_path = "/".join((self.root_path + self.files.iloc[0, 5]).split("/")[:-1]) + "/"
        source_path = "/".join((self.root_path + self.files.iloc[0, 6]).split("/")[:-1]) + "/"
        target_path = "/".join((self.root_path + self.files.iloc[0, 7]).split("/")[:-1]) + "/"

        return [ct_path, depth_path, binary_path, center_path, source_path], target_path


if __name__ == "__main__":

    root_dir = "/home/baumgartner/sgutwein84/container/training_data/training/"
    save_path = "/home/baumgartner/sgutwein84/container/3D-UNet/saved_models/"
    csv_dir = root_dir + "csv_files.xlsx"

    ds = DoseDataset(root_dir, csv_dir, patch_size=32)
    for i in range(110):
        train, target = ds[i]
        print(train.shape, target.shape)
