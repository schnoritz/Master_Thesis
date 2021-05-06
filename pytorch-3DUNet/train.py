import os
from dataset import DoseDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Dose3DUNET
import torch
import torch.nn as nn

def train(UNET, train_loader, device):

    for batch_idx, (data, target) in enumerate(train_loader):
        
        data.to(device)
        target.to(device)
        print("Input Shape: ", data.shape)
        
        pred = UNET(data)
        print("Prediction Shape: ", pred.shape)
    
def get_train_test_sets(dataset, train_fraction):

    num = len(dataset)

    train_n, test_n = int(np.ceil(num*train_fraction)), int(np.floor(num*(1-train_fraction)))

    train_set, test_set = torch.utils.data.random_split(my_dataset, [train_n, test_n])

    return train_set, test_set


if __name__ == "__main__":
    
    root_dir = "/home/baumgartner/sgutwein84/container/training_data/training/"
    csv_dir = root_dir + "csv_files.xls"
    patch_size = 32
    train_fraction = 0.8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} as calculation device!")
    
    my_dataset = DoseDataset(root_dir, csv_dir, patch_size=patch_size)

    train_set, test_set = get_train_test_sets(my_dataset, train_fraction)

    train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=True)

    my_UNET = Dose3DUNET().double().to(device)
    print(my_UNET)

    loss_func = nn.CrossEntropyLoss()

    train(my_UNET, train_loader, device=device)


