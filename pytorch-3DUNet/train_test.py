import os
from dataset import SubjectDataset, setup_loaders
import numpy as np
import matplotlib.pyplot as plt
from model import Dose3DUNET
import torch
from torch import optim
import torch.nn as nn
import argparse
import utils_test
import pickle
import dataqueue
from time import time
import sys
from tqdm import tqdm

# val
from torchvision.transforms import Normalize
from torch.utils.tensorboard import SummaryWriter


def parse():

    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('num_patches', type=int, metavar='',
                        help='Number of patches')

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


def get_loss(unet, criterion, queue, device):

    unet.eval()
    with torch.no_grad():

        loss = 0
        patches = 0
        exit = False

        for (input_patches, target_patches) in queue:
            for (input_batch, target_batch) in zip(input_patches, target_patches):

                patches += input_batch.shape[0]

                input_batch, target_batch = get_training_data(
                    input_batch, target_batch, device)

                pred = unet(input_batch)
                loss += criterion(pred, target_batch).item()

                if patches >= val_patches:
                    exit = True
                    break

            if exit == True:
                break

    unet.train()

    return loss/patches


def checkpoint(train_val_sets, train_state, curr_patches, num_patches, unet, criterion, device, batch_size, patch_size):

    print("Checkpoint reached!")

    print(
        f"---- Network has seen {train_state['start_patch_num'] + curr_patches}/{train_state['start_patch_num'] + num_patches} patches ----"
    )

    train_loss_queue = dataqueue.DataQueue(
        segment_list=train_val_sets[0],
        batch_size=batch_size,
        segments_per_queue=spq_loss,
        patch_size=patch_size,
        patches_per_segment=pps_loss
    )

    val_loss_queue = dataqueue.DataQueue(
        segment_list=train_val_sets[1],
        batch_size=batch_size,
        segments_per_queue=spq_loss,
        patch_size=patch_size,
        patches_per_segment=pps_loss
    )

    train_loss = get_loss(unet, criterion, train_loss_queue, device)

    val_loss = get_loss(unet, criterion, val_loss_queue, device)

    print(
        f"Train Loss is: {np.round(train_loss,4)} after {curr_patches} Patches")
    print(
        f"Validation Loss is:  {np.round(val_loss,4)} after {curr_patches} Patches\n")

    sys.stdout.flush()

    writer.add_scalar('Loss/train', train_loss, curr_patches)
    writer.add_scalar('Loss/validation', val_loss, curr_patches)
    writer.flush()

    return train_loss, val_loss


def train(train_val_sets, train_state, num_patches, train_queue, criterion, device, save_dir, batch_size, patch_size):

    unet = train_state['UNET']
    optimizer = train_state['optimizer']
    epochs = train_state['epochs']

    print("Start Training!\n")

    curr_patches = 0
    num = 1
    train_loss = 0
    exit = False
    early_stopping = False

    while curr_patches < num_patches:

        for (train_patches, target_patches) in train_queue:
            for (train_batch, target_batch) in tqdm(zip(train_patches, target_patches)):

                if curr_patches >= num_patches:
                    exit = True
                    break

                train_batch, target_batch = get_training_data(
                    train_batch, target_batch, device)

                dose_pred = unet(train_batch)

                loss = criterion(dose_pred, target_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                curr_patches += train_batch.shape[0]
                print(curr_patches)
                sys.stdout.flush()

                if curr_patches >= num*checkpoint_num:

                    # hier training und validation loss neu berechnen z.B. 5000 patches aus train
                    train_loss, val_loss = checkpoint(
                        train_val_sets,
                        train_state,
                        curr_patches,
                        num_patches,
                        unet,
                        criterion,
                        device,
                        batch_size,
                        patch_size,
                    )

                    epochs.append({
                        "num_patches": curr_patches+train_state['start_patch_num'],
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "model_num": train_state['model_num'] + num
                    })

                    if len(epochs) >= 21:
                        if all([i > epochs[-21]['val_loss'] for i in [x['val_loss'] for x in epochs[-20:]]]):
                            early_stopping = True
                            break

                    save = utils_test.check_improvement(epochs, top_k=top_k)

                    if save:
                        utils_test.save_model(
                            model=unet,
                            optimizer=optimizer,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            save_dir=save_dir,
                            patches=train_state['start_patch_num'] +
                            curr_patches,
                            save=save,
                            epochs=epochs,
                            model_num=num
                        )

                    num += 1

            if early_stopping == True:
                break

            if exit == True:
                break

        if exit == True or early_stopping == True:
            break

        sys.stdout.flush()

    if exit == True:
        print(
            f"Network has seen: {train_state['start_patch_num'] + num_patches} Patches!")

    if early_stopping == True:
        print(
            f"Network has seen: {train_state['start_patch_num'] + num_patches} Patches and was stopped due to no improvement over 20 validation steps!")

    sys.stdout.flush()

    return epochs


def setup_training(
    num_patches,
    batch_size,
    patch_size,
    save_dir,
    train_fraction,
    data_path,
    use_gpu,
    pretrained_model
):

    global checkpoint_num
    global top_k
    global val_patches
    checkpoint_num = 1E5
    val_patches = 5E3
    top_k = 5

    global spq_loss
    global pps_loss

    spq_loss = 10
    pps_loss = 500

    spq_train = 10
    pps_train = 5000

    device = utils_test.define_calculation_device(use_gpu)

    # subject_list = SubjectDataset(data_path, sampling_scheme="beam")
    if data_path[-1] != "/":
        data_path += "/"

    subject_list = [data_path +
                    x for x in os.listdir(data_path) if not x.startswith(".")]

    train_set, val_set = dataqueue.get_train_val_sets(
        subject_list, train_fraction=train_fraction)

    # hier segments_per_queue und patches_per_segment hochscruaben / runterschrauben
    train_queue = dataqueue.DataQueue(
        segment_list=train_set,
        batch_size=batch_size,
        segments_per_queue=spq_train,
        patch_size=patch_size,
        patches_per_segment=pps_train
    )

    my_UNET = Dose3DUNET().float()
    if torch.cuda.device_count() > 1:
        my_UNET = nn.DataParallel(my_UNET)
        print(f"Using {torch.cuda.device_count()} GPU's")

    criterion = utils_test.RMSELoss()
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
            'start_patch_num': state['patches'],
            'epochs': state['epochs'],
            'model_num': state['model_num']
        }

    else:
        train_state = {
            'UNET': my_UNET,
            'optimizer': optimizer,
            'start_patch_num': 0,
            'epochs': [],
            'model_num': 0
        }

    if device.type == 'cuda':
        train_state['UNET'] = train_state['UNET'].to(device)

    print(
        f"Training-Data shape: [{batch_size} ,5 , {patch_size}, {patch_size}, {patch_size}]\n")

    sys.stdout.flush()

    losses = train(
        train_val_sets=(train_set, val_set),
        train_state=train_state,
        num_patches=num_patches,
        train_queue=train_queue,
        save_dir=save_dir,
        criterion=criterion,
        device=device,
        batch_size=batch_size,
        patch_size=patch_size
    )

    val_losses = [losses[i]['val_loss'] for i in range(len(losses))]
    train_losses = [losses[i]['train_loss'] for i in range(len(losses))]
    patches = [losses[i]['num_patches'] for i in range(len(losses))]
    plt.plot(patches, val_losses)
    plt.plot(patches, train_losses)
    plt.yscale("log")
    plt.show()
    plt.savefig("/home/baumgartner/sgutwein84/container/logs/loss.png")

    fout = open(save_dir + "epochs_losses.pkl", "wb")
    pickle.dump(losses, fout)
    fout.close()


if __name__ == "__main__":

    global writer
    NAME = f"/home/baumgartner/sgutwein84/container/runs/model_{time()}"
    writer = SummaryWriter(NAME)

    # args = parse()
    # setup_training(
    #     num_patches=args.num_patches,
    #     batch_size=args.batch_size,
    #     patch_size=args.patch_size,
    #     train_fraction=args.train_fraction,
    #     data_path=args.root_dir,
    #     save_dir=args.save_dir,
    #     use_gpu=args.use_gpu,
    #     pretrained_model=args.pretrained_model
    # )

    setup_training(
        num_patches=200000,
        batch_size=16,
        patch_size=32,
        save_dir="/home/baumgartner/sgutwein84/container/pytorch-3DUNet/saved_models/",
        train_fraction=0.9,
        data_path="/home/baumgartner/sgutwein84/container/prostate_training_data",
        use_gpu=True,
        pretrained_model="False"
    )

    writer.close()
