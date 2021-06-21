import os
from numpy.lib.shape_base import get_array_wrap
# from torch._C import double, float

from torchio.data import queue
from dataset import SubjectDataset, setup_loaders
import numpy as np
import matplotlib.pyplot as plt
from model import Dose3DUNET
import torch
import torch.nn as nn
from torch import optim
import argparse
import torchio as tio
import utils
import pickle

# test
from torchvision.transforms import Normalize


def parse():

    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('num_epochs', type=int, metavar='',
                        help='Number of epochs')

    parser.add_argument('batch_size', type=int, metavar='',
                        help='Number of samples for one batch')

    parser.add_argument('patch_size', type=int, metavar='',
                        help='Size of one 3D-Patch: [patchsize, patchsize, patchsize]')

    parser.add_argument('--train_fraction', type=float, metavar='', default=0.9,
                        help='percent used for for training 0=0, 1=100%')

    parser.add_argument(
        '--root_dir', type=str, default="/home/baumgartner/sgutwein84/container/training_data/training/")

    parser.add_argument(
        '--save_dir', type=str, default="/home/baumgartner/sgutwein84/container/3D-UNet/saved_models/")

    parser.add_argument('--use_gpu', type=bool, metavar='', default=True,
                        help='Number of samples for one batch')

    args = parser.parse_args()

    return args


def get_training_data(batch, device):

    masks = batch["trainings_data"]['data']
    true_dose = batch["target_data"]['data']
    masks = masks.float()
    true_dose = true_dose.float()

    if device.type == 'cuda':
        masks = masks.cuda()
        true_dose = true_dose.cuda()

        masks.to(device)
        true_dose.to(device)

    return masks, true_dose


def validate(unet, criterion, test_loader, device):

    with torch.no_grad():

        val_loss = 0
        for num, batch in enumerate(test_loader):

            masks = batch["trainings_data"]['data']
            true_dose = batch["target_data"]['data']
            masks = masks.float()
            true_dose = true_dose.float()

            if device.type == 'cuda':
                masks = masks.cuda()
                true_dose = true_dose.cuda()

                masks.to(device)
                true_dose.to(device)

            pred = unet(masks)

            val_loss += criterion(pred, true_dose).item()

    return val_loss/num


def train(unet, num_epochs, train_loader, test_loader, optimizer, criterion, device, save_dir):

    print("Start Training!\n")
    total_patches = 0
    epochs = []
    for epoch in range(num_epochs):

        print(f"---- Epoch {epoch+1}/{num_epochs} ----")

        train_loss = 0
        for num, batch in enumerate(train_loader):

            masks, true_dose = get_training_data(batch, device)

            if torch.isnan(torch.sum(masks)) or torch.isnan(torch.sum(true_dose)) or torch.isinf(torch.sum(masks)) or torch.isinf(torch.sum(true_dose)):
                print('invalid input detected at iteration ', num)
                print(masks.max(), masks.min(),
                      true_dose.max(), true_dose.min())

            dose_pred = unet(masks)

            loss = criterion(dose_pred, true_dose)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print(f"Epoch Loss is: {loss.item()}")
            total_patches += masks.shape[0]

        train_loss = train_loss/num
        test_loss = validate(unet, criterion, test_loader, device)
        print(f"Train Loss is: {np.round(train_loss,4)}")
        print(f"Test Loss is:  {np.round(test_loss,4)}\n")

        epochs.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "test_loss": test_loss
        })

        save = utils.check_improvement(epochs, top_k=10)

        if save:
            utils.save_model(
                model=unet,
                optimizer=optimizer,
                train_loss=train_loss,
                test_loss=test_loss,
                save_dir=save_dir,
                epoch=epoch,
                save=save
            )

    print(f"Network has seen: {total_patches} Patches!")

    return epochs


def setup_training(
    num_epochs,
    batch_size,
    patch_size,
    save_dir,
    train_fraction,
    data_path,
    use_gpu
):

    device = utils.define_calculation_device(use_gpu)

    subject_list = SubjectDataset(data_path, sampling_scheme="beam")
    print(f'Number of Segments: {len(subject_list)}')

    train_loader, test_loader = setup_loaders(
        subject_list=subject_list,
        batch_size=batch_size,
        patch_size=patch_size,
        train_fraction=train_fraction
    )

    my_UNET = Dose3DUNET().float()

    if device.type == 'cuda':
        my_UNET.cuda().to(device)

    criterion = utils.RMSELoss()
    optimizer = optim.Adam(my_UNET.parameters(), 10E-4, (0.9, 0.99), 10E-8)

    print(
        f"Training-Data shape: [{batch_size} ,5 ,{patch_size},{patch_size},{patch_size}]")

    losses = train(
        unet=my_UNET,
        num_epochs=num_epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        save_dir=save_dir,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    test_losses = [losses[i]["test_loss"] for i in range(len(losses))]
    train_losses = [losses[i]["train_loss"] for i in range(len(losses))]
    plt.plot(test_losses)
    plt.plot(train_losses)
    plt.yscale("log")
    plt.show()
    plt.savefig("/home/baumgartner/sgutwein84/container/logs/loss.png")

    fout = open(save_dir + "epochs_losses.pkl", "wb")
    pickle.dump(losses, fout)
    fout.close()


if __name__ == "__main__":

    # args = parse()
    # setup_training(
    #     num_epochs=args.num_epochs,
    #     batch_size=args.batch_size,
    #     patch_size=args.patch_size,
    #     train_fraction=args.train_fraction,
    #     data_path=args.root_dir,
    #     save_dir=args.save_dir,
    #     use_gpu=args.use_gpu
    # )

    setup_training(
        num_epochs=10,
        batch_size=2,
        patch_size=32,
        save_dir="/home/baumgartner/sgutwein84/container/pytorch-3DUNet/saved_models/",
        train_fraction=0.9,
        data_path="/home/baumgartner/sgutwein84/container/training_data20210620",
        use_gpu=True
    )
