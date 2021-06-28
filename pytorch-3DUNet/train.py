import os
from dataset import SubjectDataset, setup_loaders
import numpy as np
import matplotlib.pyplot as plt
from model import Dose3DUNET
import torch
from torch import optim
import argparse
import utils
import pickle
import dataqueue
from time import time
import sys

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

    parser.add_argument(
        '--use_gpu', type=bool, metavar='', default=True, help='Number of samples for one batch')

    parser.add_argument(
        '--pretrained_model', type=str, metavar='', default="False", help='specify path to pretrained model')

    args = parser.parse_args()

    return args


def get_training_data(train, target, device):

    train = train.float()
    target = target.float()

    if device.type == 'cuda':
        train = train.cuda()
        target = target.cuda()

        train = train.to(device)
        target = target.to(device)

    return train, target


def validate(unet, criterion, test_queue, device):

    with torch.no_grad():

        val_loss = 0
        num = 0
        for (test_patches, target_patches) in test_queue:
            for (test_batch, target_batch) in zip(test_patches, target_patches):
                num += 1
                test_batch.float()
                target_batch.float()

                if device.type == 'cuda':
                    test_batch = test_batch.cuda()
                    target_batch = target_batch.cuda()

                    test_batch.to(device)
                    target_batch.to(device)

                pred = unet(test_batch)

                val_loss += criterion(pred, target_batch).item()

    return val_loss/num


def train(train_state, num_epochs, train_queue, test_queue, criterion, device, save_dir):

    unet = train_state['UNET']
    optimizer = train_state['optimizer']
    epochs = train_state['epochs']

    print("Start Training!\n")
    total_patches = 0

    for epoch in range(train_state['starting_epoch'], train_state['starting_epoch'] + num_epochs):

        print(
            f"---- Epoch {epoch+1}/{train_state['starting_epoch'] + num_epochs} ----")

        train_loss = 0
        num = 0
        for (train_patches, target_patches) in train_queue:
            for (train_batch, target_batch) in zip(train_patches, target_patches):
                num += 1

                train_batch, target_batch = get_training_data(
                    train_batch, target_batch, device)

                dose_pred = unet(train_batch)

                loss = criterion(dose_pred, target_batch)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(f"Epoch Loss is: {loss.item()}")
                total_patches += train_batch.shape[0]

        train_loss = train_loss/num
        test_loss = validate(unet, criterion, test_queue, device)

        print(
            f"Train Loss is: {np.round(train_loss,4)} after {total_patches} Patches")
        print(
            f"Test Loss is:  {np.round(test_loss,4)} after {total_patches} Patches\n")

        sys.stdout.flush()

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
                save=save,
                epochs=epochs
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
    use_gpu,
    pretrained_model
):

    device = utils.define_calculation_device(use_gpu)

    # subject_list = SubjectDataset(data_path, sampling_scheme="beam")
    if data_path[-1] != "/":
        data_path += "/"

    subject_list = [data_path +
                    x for x in os.listdir(data_path) if not x.startswith(".")]

    print(f'Number of Segments: {len(subject_list)}')

    # train_loader, test_loader = setup_loaders(
    #     subject_list=subject_list,
    #     batch_size=batch_size,
    #     patch_size=patch_size,
    #     train_fraction=train_fraction
    # )

    train_set, test_set = dataqueue.get_train_test_sets(
        subject_list, train_fraction=train_fraction)

    train_queue = dataqueue.DataQueue(
        segment_list=train_set,
        batch_size=batch_size,
        segments_per_queue=5,
        patch_size=patch_size,
        patches_per_segment=250
    )

    test_queue = dataqueue.DataQueue(
        segment_list=test_set,
        batch_size=batch_size,
        segments_per_queue=5,
        patch_size=patch_size,
        patches_per_segment=250
    )

    my_UNET = Dose3DUNET().float()
    criterion = utils.RMSELoss()
    optimizer = optim.Adam(my_UNET.parameters(), 10E-4, (0.9, 0.99), 10E-8)

    if pretrained_model != "False":
        model_name = pretrained_model.split("/")[-1]
        print(f"\nUsing pretrained Model: {model_name}\n")
        if torch.cuda.is_available():
            state = torch.load(pretrained_model)
        else:
            state = torch.load(
                pretrained_model, map_location=torch.device('cpu'))
            my_UNET.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])

        train_state = {
            'UNET': my_UNET,
            'optimizer': optimizer,
            'starting_epoch': state['epoch'],
            'epochs': state['epochs']
        }

    else:
        train_state = {
            'UNET': my_UNET,
            'optimizer': optimizer,
            'starting_epoch': 0,
            'epochs': []
        }

    if device.type == 'cuda':
        train_state['UNET'] = train_state['UNET'].to(device)

    print(
        f"Training-Data shape: [{batch_size} ,5 ,{patch_size},{patch_size},{patch_size}]\n")

    sys.stdout.flush()

    losses = train(
        train_state=train_state,
        num_epochs=num_epochs,
        train_queue=train_queue,
        test_queue=test_queue,
        save_dir=save_dir,
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

    args = parse()
    setup_training(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        train_fraction=args.train_fraction,
        data_path=args.root_dir,
        save_dir=args.save_dir,
        use_gpu=args.use_gpu,
        pretrained_model=args.pretrained_model
    )

    # setup_training(
    #     num_epochs=10,
    #     batch_size=16,
    #     patch_size=32,
    #     save_dir="/home/baumgartner/sgutwein84/container/pytorch-3DUNet/saved_models/",
    #     train_fraction=0.9,
    #     data_path="/home/baumgartner/sgutwein84/container/training_data20210619",
    #     use_gpu=True,
    #     pretrained_model="False"
    # )
