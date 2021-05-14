import os
from dataset_old import DoseDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Dose3DUNET
import torch
import torch.nn as nn
from torch import optim
from time import time
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('epochs', type=int, metavar='',
                        help='Number of epochs')

    parser.add_argument('batch_size', type=int, metavar='',
                        help='Number of samples for one batch')

    parser.add_argument('patch_size', type=int, metavar='',
                        help='Size of one 3D-Patch: [patchsize, patchsize, patchsize]')

    parser.add_argument('--train_fraction', type=float, metavar='', default=0.9,
                        help='percent used for for training 0=0, 1=100%')

    parser.add_argument(
        '--rootdir', type=str, default="/home/baumgartner/sgutwein84/container/training_data/training/")

    parser.add_argument(
        '--savepath', type=str, default="/home/baumgartner/sgutwein84/container/3D-UNet/saved_models/")
        
    parser.add_argument('--use_gpu', type=bool, metavar='', default=True, 
                        help='Number of samples for one batch')

    args = parser.parse_args()

    return args


def train(UNET, epochs, train_loader, test_loader, optimizer, criterion, device, save_dir):

    print("Start Training!")

    train_losses = []
    val_losses = []
    for epoch in range(epochs):

        print(f"Epoch {epoch+1}/{epochs}")

        train_loss = 0
        for batch_idx, (masks, true_dose) in enumerate(train_loader):
        #for i in range(4):

        # masks = torch.randn((2, 5, 16, 16, 16)).double()
        # true_dose = torch.randn((2, 1, 16, 16, 16)).double()
            
            if device.type == 'cuda':
                masks = masks.cuda()
                true_dose = true_dose.cuda()

                masks.to(device)
                true_dose.to(device)
    
            dose_pred = UNET(masks)

            loss = criterion(dose_pred, true_dose)

            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            print(f"Loss is: {np.round(loss.item(),4)}")  

        train_loss = train_loss/batch_idx
        val_loss = validate(UNET, criterion)
        train_losses.append(train_loss)

        if len(losses) > 1:
            if val_loss < val_losses[-1]:
                torch.save(UNET.state_dict(), save_dir + f"test_model.pth")
        
        else:
            torch.save(UNET.state_dict(), save_dir + f"test_model.pth")

        val_losses.append(val_loss)

    return losses

def validate(NET, criterion):

    with torch.no_grad():
        val_loss
        for batch_idx, (masks, target) in enumerate(test_loader):

            pred = NET(masks)
            
            val_loss += criterion(pred, target).item()

        
    return val_loss/batch_idx







class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

    
def get_train_test_sets(dataset, train_fraction):

    num = len(dataset)

    train_n, test_n = int(np.ceil(num*train_fraction)), int(np.floor(num*(1-train_fraction)))

    train_set, test_set = torch.utils.data.random_split(dataset, [train_n, test_n])

    return train_set, test_set

def define_calculation_device(use_gpu):

    if use_gpu:
        if torch.cuda.is_available():
            print("Using CUDA!")
            device = torch.device('cuda')
        else:
            print("Using CPU!")
            device = torch.device('cpu')
    else:
        print("Using CPU!")
        device = torch.device('cpu')

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
        
    return device


def setup_training(epochs, batch_size, patch_size, save_dir, train_fraction, root_dir, use_gpu):

    csv_dir = root_dir + "csv_files.xls"

    device = define_calculation_device(use_gpu)

    my_dataset = DoseDataset(root_dir, csv_dir, patch_size=patch_size)

    train_set, test_set = get_train_test_sets(my_dataset, train_fraction)

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True)

    my_UNET = Dose3DUNET().double()
    
    if device.type == 'cuda':
        my_UNET.cuda().to(device)

    criterion = RMSELoss()
    optimizer = optim.Adam(my_UNET.parameters(), 10E-4, (0.9, 0.99), 10E-8)

    losses = train(my_UNET, epochs, train_loader, test_laoder, save_dir=save_dir,
                   optimizer=optimizer, criterion=criterion, device=device)

    plt.plot(losses)
    plt.show()
    plt.savefig("/home/baumgartner/sgutwein84/container/logs/loss.png")

if __name__ == "__main__":

    #args = parse()
    #setup_training(args.epochs, args.batch_size, args.patch_size, args.train_fraction, args.root_dir, args.save_dir, args.use_gpu)

    setup_training(10, 16, 32, "/home/baumgartner/sgutwein84/container/3D-UNet/saved_models/",
                   0.9, "/home/baumgartner/sgutwein84/container/training_data/training/", True)