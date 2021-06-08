import os
# from torch._C import double, float

from torchio.data import queue
from dataset import SubjectDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Dose3DUNET
import torch
import torch.nn as nn
from torch import optim
import argparse
import torchio as tio
from utils import RMSELoss, Color
import pickle


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
        '--rootdir', type=str, default="/home/baumgartner/sgutwein84/container/training_data/training/")

    parser.add_argument(
        '--savepath', type=str, default="/home/baumgartner/sgutwein84/container/3D-UNet/saved_models/")

    parser.add_argument('--use_gpu', type=bool, metavar='', default=True,
                        help='Number of samples for one batch')

    args = parser.parse_args()

    return args


def train(unet, num_epochs, train_loader, test_loader, optimizer, criterion, device, save_dir):

    print("Start Training!")
    total_patches = 0
    epochs = []
    for epoch in range(num_epochs):

        print(Color.BOLD + f"Epoch {epoch+1}/{num_epochs}" + Color.END)

        train_loss = 0
        for num, batch in enumerate(train_loader):

            masks = batch["trainings_data"]['data']
            true_dose = batch["target_data"]['data']
            masks = masks.float()
            true_dose = true_dose.float()

            total_patches += masks.shape[0]

            if device.type == 'cuda':
                masks = masks.cuda()
                true_dose = true_dose.cuda()

                masks.to(device)
                true_dose.to(device)

            dose_pred = unet(masks)
            print(dose_pred.shape)

            loss = criterion(dose_pred, true_dose)

            train_loss += loss.item()
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch Loss is: {loss.item()}")

        train_loss = train_loss/num
        test_loss = validate(unet, criterion, test_loader, device)
        print(f"Train Loss is: {np.round(train_loss,4)}")
        print(f"Test Loss is:  {np.round(test_loss,4)}\n")

        epochs.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "test_loss": test_loss
        })

        save = check_improvement(epochs, top_k=5)
        if save:
            torch.save(
                unet.state_dict(),
                save_dir + f"UNET_epoch{epoch+1}.pth"
            )
            if type(save) == int:
                os.remove(
                    save_dir + f"UNET_epoch{save}.pth")

    print(f"Network has seen: {total_patches} Patches!")

    return epochs


def check_improvement(epochs, top_k=5):

    curr_epoch = epochs[-1]
    epochs = sorted(epochs, key=lambda k: k['test_loss'])
    if epochs.index(curr_epoch) < top_k:
        if len(epochs) > top_k:
            return epochs[top_k]["epoch"]
        else:
            return True
    else:
        return False


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


def get_train_test_sets(dataset, train_fraction):

    num = len(dataset)

    train_n, test_n = int(np.ceil(num*train_fraction)
                          ), int(np.floor(num*(1-train_fraction)))

    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_n, test_n])

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
        print("Device: " + torch.cuda.get_device_name(0))
        # print('Memory Usage:')
        # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        # print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

    return device


def setup_training(
    num_epochs,
    batch_size,
    patch_size,
    save_dir,
    train_fraction,
    data_path,
    use_gpu
):

    device = define_calculation_device(use_gpu)

    SubjectList = SubjectDataset(data_path)
    print(f'Number of Segments: {len(SubjectList)}')

    train_set, test_set = get_train_test_sets(SubjectList, train_fraction)

    sampler = tio.data.WeightedSampler(
        patch_size=patch_size, probability_map='sampling_map')

    # tbd
    samples_per_volume = 512
    queue_length = samples_per_volume*2

    train_queue = tio.Queue(
        train_set,
        queue_length,
        samples_per_volume,
        sampler,
        shuffle_patches=True,
        # shuffle_subjects=True
    )

    test_queue = tio.Queue(
        test_set,
        queue_length,
        samples_per_volume,
        sampler,
        shuffle_patches=True,
        # shuffle_subjects=True
    )

    train_loader = DataLoader(
        dataset=train_queue,
        batch_size=batch_size,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_queue,
        batch_size=batch_size,
        num_workers=2
    )

    my_UNET = Dose3DUNET().float()

    if device.type == 'cuda':
        my_UNET.cuda().to(device)

    criterion = RMSELoss()
    optimizer = optim.Adam(my_UNET.parameters(), 10E-5, (0.9, 0.99), 10E-8)

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
    plt.show()
    plt.savefig("/home/baumgartner/sgutwein84/container/logs/loss.png")

    fout = open(save_dir + "epochs_losses.pkl", "wb")
    pickle.dump(losses, fout)
    fout.close()


if __name__ == "__main__":

    # args = parse()
    # setup_training(args.num_epochs, args.batch_size, args.patch_size, args.train_fraction, args.root_dir, args.save_dir, args.use_gpu)

    setup_training(
        num_epochs=100,
        batch_size=64,
        patch_size=32,
        save_dir="/home/baumgartner/sgutwein84/container/pytorch-3DUNet/saved_models/",
        train_fraction=0.9,
        data_path="/home/baumgartner/sgutwein84/container/training_data20210608",
        use_gpu=True
    )
