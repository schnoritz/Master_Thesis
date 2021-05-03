import torch
from torch.utils.data import Dataset
import numpy as np
import os

class DoseDataset(Dataset):
    def __init__(self, mask_dir, dose_dir, transform=None):
        self.mask_dir = mask_dir
        self.dose_dir = dose_dir
        self.transform = transform
        self.mask_dirs = os.listdir(mask_dir)
        self.binary_masks = os.listdir(mask_dir + '/binary')
        self.SD_masks = os.listdir(mask_dir + '/source_distance')
        self.CD_masks = os.listdir(mask_dir + '/center_distance')
        self.depth_masks = os.listdir(mask_dir + '/radio_depth')
        self.ct_mask = os.listdir(mask_dir + '/ct')



def test():
    mask_dir = "/Users/simongutwein/home/baumgartner/sgutwein84/training_data/training"
    dose_dir = "/Users/simongutwein/home/baumgartner/sgutwein84/training_data/target"
    dataset = DoseDataset(mask_dir, dose_dir)
    print(dataset)

if __name__ == "__main__":
    test()
    pass

# binary_mask = torch.randn((32, 32, 32))
# sd_mask = torch.randn((32, 32, 32))
# cd_mask = torch.randn((32, 32, 32))
# ct_mask = torch.randn((32, 32, 32))
# depth_mask = torch.randn((32, 32, 32))

# patch = torch.stack((binary_mask, sd_mask, cd_mask, ct_mask, depth_mask), 0)
# print(patch.shape)

# batch = torch.stack((patch, patch, patch, patch), 0)
# print(batch.shape)
