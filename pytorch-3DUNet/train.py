import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from model import Dose3DUNET
import torch
from torch import optim
import torch.nn as nn
import argparse
import utils
from losses import RMSELoss
import pickle
import dataqueue
from time import time
import sys
from gamma_sample import gamma_sample
import numpy as np


from torch.utils.tensorboard import SummaryWriter


def parse():

    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('-num_patches', type=int, metavar='',
                        help='Number of patches')

    parser.add_argument('-batch_size', type=int, metavar='',
                        help='Number of samples for one batch')

    parser.add_argument('-patch_size', type=int, metavar='',
                        help='Size of one 3D-Patch: [patchsize, patchsize, patchsize]')

    parser.add_argument('-experiment_name', type=str,
                        metavar='', help='name for the experiment')

    parser.add_argument('-validation_segments', nargs='+',
                        help='set segments used for validation')

    parser.add_argument(
        '--root_dir', type=str, default="/home/baumgartner/sgutwein84/container/prostate_training_data/")

    parser.add_argument(
        '--save_dir', type=str, default="")

    parser.add_argument(
        '--use_gpu', type=bool, metavar='', default=True, help='Number of samples for one batch')

    parser.add_argument(
        '--pretrained_model', type=str, metavar='', default="False", help='specify path to pretrained model')

    parser.add_argument(
        '--learning_rate', type=float, metavar='', default=0.00001, help='specify the learning rate')

    parser.add_argument(
        '--writer', type=str, metavar='', default="None", help='specify the writer')

    args = parser.parse_args()

    return args


def validate(unet, criterion, validation_queue, device):

    unet.eval()
    curr = 0
    with torch.no_grad():

        loss = 0
        for num, (input_batch, target_batch) in enumerate(validation_queue):
            curr += input_batch.shape[0]
            progressBar(curr, len(validation_queue), "Validation Step")
            sys.stdout.flush()
            input_batch, target_batch = utils.get_training_data(
                input_batch, target_batch, device)

            pred = unet(input_batch)
            loss += criterion(pred, target_batch).item()

    unet.train()
    del input_batch
    del target_batch
    del pred

    return loss / num


def get_volume_prediction(net, device, curr_patches):

    torch.cuda.empty_cache()

    with torch.no_grad():
        net.eval()

        for seg in test_segments:

            masks = torch.load(f"{seg}/training_data.pt")
            masks = masks[:, 128:128+256, 128:128+256, 25:25+16]
            masks = torch.unsqueeze(masks, 0)
            masks.to(device)

            dose = torch.load(f"{seg}/target_data.pt")

            pred = net(masks)

            pred = pred.cpu().detach().numpy()
            pred = pred.squeeze()

            masks = masks.cpu().detach().numpy()
            masks = masks.squeeze()

            dose = dose.squeeze()
            dose = dose[128:128+256, 128:128+256, 25:25+16]

            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            ax[0].imshow(masks[0, :, :, 8])
            ax[1].imshow(dose[:, :, 8])
            ax[2].imshow(pred[:, :, 8])
            ax[3].imshow(dose[:, :, 8]-pred[:, :, 8])

            del masks
            torch.cuda.empty_cache()

            writer.add_figure(f'{seg}/prediction',
                              fig, global_step=curr_patches)
            writer.flush()

            plt.close(fig)

    net.train()


def progressBar(current, total, description, barLength=50):

    percent = int(current * 100 / total)
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print(
        f'Progress: [{arrow}{spaces}] {percent}% [{current}/{total}] // {description}', end="\n")
    sys.stdout.flush()


def train(train_state, num_patches, train_queue, validation_queue, criterion, device, save_dir, batch_size, patch_size):

    unet = train_state['UNET']
    unet = unet.to(device)
    optimizer = train_state['optimizer']
    utils.optimizer_to_device(optimizer, device)
    epochs = train_state['epochs']

    print("Start Training!\n")

    curr_patches = train_state['start_patch_num']
    num_patches = num_patches + curr_patches
    generation = train_state['model_generation']

    while curr_patches < num_patches:

        generation += 1
        torch.cuda.empty_cache()
        train_queue.load_queue()
        train_loss = 0
        generation_patches = 0

        print("Generation : ", generation)
        for num, (train_batch, target_batch) in enumerate(train_queue):

            train_batch, target_batch = utils.get_training_data(
                train_batch,
                target_batch,
                device
            )

            dose_pred = unet(train_batch)
            torch.cuda.empty_cache()

            loss = criterion(dose_pred, target_batch)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_patches += train_batch.shape[0]
            generation_patches += train_batch.shape[0]

            progressBar(generation_patches, len(train_queue), "Training Step")

        print("\n")
        train_loss /= num

        validation_loss = validate(unet, criterion, validation_queue, device)
        gamma_values = []
        for seg in ["p9_35", "p9_11", "p7_34", "p7_1", "p8_9", "p8_20", "p5_42", "p5_11"]:
            gamma_values.append(gamma_sample(
                unet, device, seg, segment_dir=data_directory))

        gammas = np.array(gamma_values)
        mean_gamma = np.round(gammas.mean(), 2)
        std_gamma = np.round(gammas.std(), 2)

        writer.add_scalar("Gamma/Mean", mean_gamma, curr_patches)
        writer.add_scalar("Gamma/STD", std_gamma, curr_patches)

        print("\n")
        print(f"Current Generation      : {generation}")
        print(f"Current Patches         : {curr_patches}")
        print(f"Current Training Loss   : {train_loss}")
        print(f"Current Validation Loss : {validation_loss}")
        print(f"Current Gamma Val       : {mean_gamma} ± {std_gamma}")
        print("\n")

        writer.add_scalar("Loss / Training Loss", train_loss, curr_patches)
        writer.add_scalar("Loss / Validation Loss",
                          validation_loss, curr_patches)

        sys.stdout.flush()

        get_volume_prediction(unet, device, curr_patches)

        epochs.append({
            "num_patches": curr_patches,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "model_generation": train_state['model_generation'] + generation
        })

        # if len(epochs) >= 36:
        #     if all([i > epochs[-36]['validation_loss'] for i in [x['validation_loss'] for x in epochs[-35:]]]):
        #         early_stopping = True
        #         break

        save = utils.check_improvement(epochs, top_k=top_k)

        if save:
            utils.save_model(
                model=unet,
                optimizer=optimizer,
                train_loss=train_loss,
                validation_loss=validation_loss,
                save_dir=save_dir,
                patches=curr_patches,
                save=save,
                epochs=epochs,
                generation=generation,
                gammas=[mean_gamma, std_gamma]
            )

    sys.stdout.flush()

    return epochs


def setup_training(
    num_patches,
    batch_size,
    patch_size,
    save_dir,
    validation_segments,
    data_path,
    use_gpu,
    pretrained_model,
    learning_rate
):

    if data_path[-1] != "/":
        data_path += "/"

    global test_segments
    test_segments = [data_path + 'p0_0', data_path + 'p10_20']

    device = utils.define_calculation_device(use_gpu)

    # subject_list = SubjectDataset(data_path, sampling_scheme="beam")

    my_UNET = Dose3DUNET().float()
    criterion = RMSELoss()

    if pretrained_model != "False":
        print(f"\nUsing pretrained Model: {pretrained_model}\n")
        state = torch.load(pretrained_model, map_location=device)
        my_UNET.load_state_dict(state['model_state_dict'])
        optimizer = optim.Adam(my_UNET.parameters(),
                               learning_rate, (0.9, 0.99), 10E-8)
        optimizer.load_state_dict(state['optimizer_state_dict'])

        if torch.cuda.device_count() > 1:
            my_UNET = nn.DataParallel(my_UNET)
            print(f"Using {torch.cuda.device_count()} GPU's")

        train_state = {
            'UNET': my_UNET,
            'optimizer': optimizer,
            'start_patch_num': state['patches'],
            'epochs': state['epochs'],
            'model_generation': state['model_generation']
        }

    else:

        if torch.cuda.device_count() > 1:
            my_UNET = nn.DataParallel(my_UNET)
            print(f"Using {torch.cuda.device_count()} GPU's")

        optimizer = optim.Adam(my_UNET.parameters(),
                               learning_rate, (0.9, 0.99), 10E-8)
        train_state = {
            'UNET': my_UNET,
            'optimizer': optimizer,
            'start_patch_num': 0,
            'epochs': [],
            'model_generation': 0
        }

    print("Validation Segments used: ", validation_segments)

    validation_segments = [data_path + x for x in validation_segments]

    subject_list = [data_path +
                    x for x in os.listdir(data_path) if not x.startswith(".") and not data_path + x in validation_segments]
    print(len(subject_list), len(validation_segments))

    # hier segments_per_queue und patches_per_segment hochscruaben / runterschrauben
    train_queue = dataqueue.DataQueue(
        segment_list=subject_list,
        batch_size=batch_size,
        segments_per_queue=spq_train,
        patch_size=patch_size,
        patches_per_segment=pps_train
    )

    validation_queue = dataqueue.ValidationQueue(
        segments=validation_segments,
        batch_size=16
    )

    print(
        f"Training-Data shape: [{batch_size} ,5 , {patch_size}, {patch_size}, {patch_size}]\n")
    print(f"Learning Rate: {learning_rate}")
    sys.stdout.flush()

    losses = train(
        train_state=train_state,
        num_patches=num_patches,
        train_queue=train_queue,
        validation_queue=validation_queue,
        save_dir=save_dir,
        criterion=criterion,
        device=device,
        batch_size=batch_size,
        patch_size=patch_size
    )

    fout = open(save_dir + "epochs_losses.pkl", "wb")
    pickle.dump(losses, fout)
    fout.close()


if __name__ == "__main__":

    args = parse()

    global top_k
    top_k = 5

    global spq_train
    global pps_train
    spq_train = 50
    pps_train = int(args.batch_size * 20)

    global data_directory
    data_directory = args.root_dir

    global writer
    if args.writer == "None":
        NAME = f"/home/baumgartner/sgutwein84/container/pytorch-3DUNet/experiments/runs/model_{args.experiment_name}_{time()}"
        print("Model information for Tensorboard saved under", NAME)
        writer = SummaryWriter(NAME)
    else:
        NAME = args.writer
        print("Model information for Tensorboard saved under", NAME)
        writer = SummaryWriter(NAME)

    setup_training(
        num_patches=args.num_patches,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        validation_segments=args.validation_segments,
        data_path=args.root_dir,
        save_dir=args.save_dir,
        use_gpu=args.use_gpu,
        pretrained_model=args.pretrained_model,
        learning_rate=args.learning_rate
    )

    writer.close()
