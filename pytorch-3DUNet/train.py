import os
from dataset_test import DoseDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Dose3DUNET
import torch
import torch.nn as nn
from torch import optim
from time import time

def train(UNET, epochs, train_loader, optimizer, criterion, device):

    print("Start Training!")
    losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        for batch_idx, (masks, true_dose) in enumerate(train_loader):
            
            if torch.cuda.is_available():
                masks = masks.cuda()
                true_dose = true_dose.cuda()
                masks.to(device)
                true_dose.to(device)
        
            start = time()
            dose_pred = UNET(masks)
            loss = criterion(dose_pred, true_dose)

            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            print(f"Loss is: {np.round(loss.item(),4)} and step took: {np.round(time()-start)} seconds.")  
        
        losses.append(epoch_loss)

    return losses

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

    
def get_train_test_sets(dataset, train_fraction):

    num = len(dataset)

    train_n, test_n = int(np.ceil(num*train_fraction)), int(np.floor(num*(1-train_fraction)))

    train_set, test_set = torch.utils.data.random_split(my_dataset, [train_n, test_n])

    return train_set, test_set


if __name__ == "__main__":
    
    root_dir = "/home/baumgartner/sgutwein84/container/training_data/training/"
    save_path = "/home/baumgartner/sgutwein84/container/3D-UNet/saved_models/"
    csv_dir = root_dir + "csv_files.xls"
    patch_size = 32
    batch_size = 10
    epochs = 10
    train_fraction = 0.9

    training=True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} as calculation device!")
    
    my_dataset = DoseDataset(root_dir, csv_dir, patch_size=patch_size)

    train_set, test_set = get_train_test_sets(my_dataset, train_fraction)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    my_UNET = Dose3DUNET().double().cuda().to(device=device)

    #print(my_UNET)

    criterion = RMSELoss()
    optimizer = optim.Adam(my_UNET.parameters(), 10E-4, (0.9, 0.99), 10E-8)

    if training:
        losses = train(my_UNET, epochs,train_loader, optimizer=optimizer, criterion=criterion, device=device)
        
        torch.save(my_UNET, save_path + "saved_3DUNet.pth")

        plt.plot(losses)
        plt.show()
        plt.savefig("/home/baumgartner/sgutwein84/container/logs/loss.png")

    if not training:

        trained_3DUNet = Dose3DUNET()
        trained_3DUNet = torch.load(save_path + "saved_3DUNet.pth")
        trained_3DUNet.eval()
        print("Model loaded")


        for nums_tests in range(20):
            for batch_idx, (masks, true_dose) in enumerate(test_loader):
                pred = trained_3DUNet(masks)

                for i in range(14):
                    
                    fig, ax = plt.subplots(1,7, figsize=(24, 4))

                    for j in range(5):
                        ax[j].imshow(np.squeeze(masks[i][j, :, :, 15].detach().numpy()))

                    ax[5].imshow(np.squeeze(pred[i][:, :, 15].detach().numpy()))
                    ax[6].imshow(np.squeeze(true_dose[i][:, :, 15].detach().numpy()))
                    plt.show()
                    plt.savefig(f"/home/baumgartner/sgutwein84/container/logs/test{nums_tests}_{i}.png")
                    
                    print("Image saved!")
                    plt.close()
