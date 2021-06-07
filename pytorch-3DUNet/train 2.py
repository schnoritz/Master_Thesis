import os

from torchio.data import queue
from dataset import SubjectDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Dose3DUNET
import torch
import torch.nn as nn
from torch import optim
#from time import time
import argparse
import torchio as tio

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
        for  num, batch in enumerate(train_loader):

            masks = batch["trainings_data"]['data']
            true_dose = batch["target_data"]['data']
            masks = masks.double()
            true_dose = true_dose.double()
            print(masks.shape)
            print(true_dose.shape)

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

        train_loss = train_loss/num
        val_loss = validate(UNET, test_loader, criterion)
        train_losses.append(train_loss)

        if len(train_losses) > 1:
            if val_loss < val_losses[-1]:
                torch.save(UNET.state_dict(), save_dir + f"test_model.pth")
        
        else:
            torch.save(UNET.state_dict(), save_dir + f"test_model.pth")

        val_losses.append(val_loss)

    return train_losses

def validate(NET, test_loader, criterion):

    with torch.no_grad():
        val_loss = []
        for num, batch in enumerate(test_loader):
            
            masks = batch["trainings_data"]['data']
            true_dose = batch["target_data"]['data']

            pred = NET(masks)
            
            val_loss += criterion(pred, true_dose).item()

    return val_loss/num

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

    
def get_train_test_sets(dataset, train_fraction):

    num = len(dataset)
    training = int(num*train_fraction)
    
    train_n, test_n = training, num-training
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


def setup_training(epochs, batch_size, patch_size, save_dir, train_fraction, data_path, use_gpu):

    device = define_calculation_device(use_gpu)

    SubjectList = SubjectDataset(data_path)
    print(len(SubjectList))

    train_set, test_set = get_train_test_sets(SubjectList, train_fraction)

    sampler = tio.data.WeightedSampler(patch_size=patch_size, probability_map='sampling_map')

    #tbd
    patch_size = 64
    samples_per_volume = 256
    queue_length = samples_per_volume*4
    batch_size = 32

    train_queue = tio.Queue(
        train_set,
        queue_length,
        samples_per_volume,
        sampler,
        shuffle_patches=True,
        #shuffle_subjects=True
    )

    test_queue = tio.Queue(
        train_set,
        queue_length,
        samples_per_volume,
        sampler,
        shuffle_patches=True,
        #shuffle_subjects=True
    )

    train_loader = DataLoader(
        dataset=train_queue, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(
        dataset=test_queue, batch_size=batch_size, num_workers=2)

    my_UNET = Dose3DUNET().double()
    
    if device.type == 'cuda':
        my_UNET.cuda().to(device)

    criterion = RMSELoss()
    optimizer = optim.Adam(my_UNET.parameters(), 10E-4, (0.9, 0.99), 10E-8)

    losses = train(my_UNET, epochs, train_loader, test_loader, save_dir=save_dir,
                   optimizer=optimizer, criterion=criterion, device=device)

    plt.plot(losses)
    plt.show()
    plt.savefig("/home/baumgartner/sgutwein84/container/logs/loss.png")

if __name__ == "__main__":

    #args = parse()
    #setup_training(args.epochs, args.batch_size, args.patch_size, args.train_fraction, args.root_dir, args.save_dir, args.use_gpu)

    setup_training(10, 16, 32, "/home/baumgartner/sgutwein84/container/3D-UNet/saved_models/",
                   0.8, "/home/baumgartner/sgutwein84/container/training", True)
