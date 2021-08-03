import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from model_UQ import Dose3DUNET
import torch
from torch import optim
import torch.nn as nn
import argparse
import utils
from losses import HeteroscedasticAleatoricLoss
import pickle
import dataqueue_threading
from time import time
import sys
from gamma_sample import gamma_sample
import numpy as np


from torch.utils.tensorboard import SummaryWriter

global TOP_K
TOP_K = 5

global SPQ_TRAIN
global PPS_TRAIN
global DATA_DIRECTORY
global LOGGER
global LOGGER_DIR
global LOGGER_SEGMENTS_LIST
global LOGGER_SEGMENTS
global LOSS_FUNCTION

LOGGER_DIR = "/mnt/qb/baumgartner/sgutwein84/logger/"
LOGGER_SEGMENTS_LIST = ["p0_0", "p8_55"]

LOSS_FUNCTION = HeteroscedasticAleatoricLoss()


def parse():

    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument(
        "-num_patches",
        type=int,
        action='store',
        required=True,
        dest='num_patches'
    )

    parser.add_argument(
        "-bs",
        type=int,
        action='store',
        required=True,
        dest='batch_size'
    )

    parser.add_argument(
        "-ps",
        type=int,
        action='store',
        required=True,
        dest='patch_size'
    )
    parser.add_argument(
        "-lr",
        type=float,
        action='store',
        required=True,
        dest='learning_rate'
    )

    parser.add_argument(
        "-exp_name",
        action='store',
        required=True,
        dest='experiment_name'
    )

    parser.add_argument(
        "-val_seg",
        required=True,
        nargs='+',
        action='store',
        dest='validation_segments'
    )

    parser.add_argument(
        "-dir",
        required=True,
        action='store',
        dest='root_dir'
    )

    parser.add_argument(
        "-save_dir",
        required=True,
        action='store',
        dest='save_dir'
    )

    parser.add_argument(
        "-gpu",
        default=True,
        action='store',
        dest='use_gpu'
    )

    parser.add_argument(
        "-pre_model",
        action='store',
        dest='pretrained_model'
    )

    parser.add_argument(
        "-logger",
        action='store',
        dest='logger'
    )

    args = parser.parse_args()

    return args


def validate(model, criterion, validation_queue, device):

    torch.cuda.empty_cache()
    model.eval()
    curr = 0
    with torch.no_grad():

        loss = 0
        for num, (input_batch, target_batch) in enumerate(validation_queue):
            curr += input_batch.shape[0]
            progressBar(curr, len(validation_queue), "Validation Step")
            sys.stdout.flush()
            input_batch, target_batch = utils.get_training_data(
                input_batch, target_batch, device)

            pred, uncertainty = model(input_batch)
            loss += criterion(pred, target_batch, uncertainty).item()

    model.train()
    del input_batch
    del target_batch
    del pred

    return loss / num


def get_volume_prediction(model, device, curr_patches, thickness=16):

    torch.cuda.empty_cache()

    with torch.no_grad():
        model.eval()

        for seg in LOGGER_SEGMENTS:

            masks = torch.load(f"{seg}/training_data.pt")
            masks = masks[:, 128:128+256, 128:128+256, 32:32+thickness]
            masks = torch.unsqueeze(masks, 0)
            masks.to(device)

            dose = torch.load(f"{seg}/target_data.pt")

            pred, uncertrainty = model(masks)

            pred = pred.cpu().detach().numpy()
            pred = pred.squeeze()

            #uncertrainty = torch.exp(uncertrainty)
            uncertrainty = uncertrainty.cpu().detach().numpy()
            uncertrainty = uncertrainty.squeeze()

            masks = masks.cpu().detach().numpy()
            masks = masks.squeeze()

            dose = dose.squeeze()
            dose = dose[128:128+256, 128:128+256, 32:32+thickness]

            fig, ax = plt.subplots(1, 5, figsize=(thickness, 4))
            ax[0].imshow(masks[0, :, :, thickness//2])
            ax[1].imshow(dose[:, :, thickness//2])
            ax[2].imshow(pred[:, :, thickness//2])
            ax[3].imshow(dose[:, :, thickness//2]-pred[:, :, thickness//2])
            ax[4].imshow(uncertrainty[:, :, thickness//2])

            del masks
            torch.cuda.empty_cache()

            LOGGER.add_figure(f'{seg}/prediction',
                              fig, global_step=curr_patches)
            LOGGER.flush()

            plt.close(fig)

    model.train()


def progressBar(current, total, description, barLength=50):

    percent = int(current * 100 / total)
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print(
        f'Progress: [{arrow}{spaces}] {percent}% [{current}/{total}] // {description}', end="\n")
    sys.stdout.flush()


def train(
    train_state,
    num_patches,
    train_queue,
    validation_queue,
    criterion,
    device,
    save_dir,
    training_parameter
):

    model = train_state['UNET']
    model.to(device)
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

            dose_pred, uncertrainty = model(train_batch)
            torch.cuda.empty_cache()

            loss = criterion(dose_pred, target_batch, uncertrainty)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_patches += train_batch.shape[0]
            generation_patches += train_batch.shape[0]

            progressBar(generation_patches, len(train_queue), "Training Step")

        print("\n")
        train_loss /= num

        validation_loss = validate(model, criterion, validation_queue, device)

        gamma_values = []
        for seg in ["p9_35", "p9_11", "p7_34", "p7_1", "p8_9", "p8_20", "p5_42", "p5_11"]:
            gamma_values.append(gamma_sample(
                model, device, seg, segment_dir=DATA_DIRECTORY, UQ=True))

        gammas = np.array(gamma_values)
        mean_gamma = np.round(gammas.mean(), 2)
        std_gamma = np.round(gammas.std(), 2)

        update_logger(mean_gamma, curr_patches, std_gamma,
                      train_loss, validation_loss)

        print_training_status(generation, curr_patches,
                              train_loss, validation_loss, mean_gamma, std_gamma)

        get_volume_prediction(model, device, curr_patches)

        epochs.append({
            "num_patches": curr_patches,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "model_generation": train_state['model_generation'] + generation
        })

        # evlt noch automated stopping nutzen und LR scheduler

        save = utils.check_improvement(epochs, top_k=TOP_K)

        if save:
            utils.save_model(
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                validation_loss=validation_loss,
                save_dir=save_dir,
                patches=curr_patches,
                save=save,
                epochs=epochs,
                generation=generation,
                gammas=[mean_gamma, std_gamma],
                training_parameter=training_parameter
            )

    sys.stdout.flush()

    return epochs


def update_logger(mean_gamma, curr_patches, std_gamma, train_loss, validation_loss):

    LOGGER.add_scalar("Gamma/Mean", mean_gamma, curr_patches)
    LOGGER.add_scalar("Gamma/STD", std_gamma, curr_patches)
    LOGGER.add_scalar("Loss / Training Loss", train_loss, curr_patches)
    LOGGER.add_scalar("Loss / Validation Loss",
                      validation_loss, curr_patches)


def print_training_status(generation, curr_patches, train_loss, validation_loss, mean_gamma, std_gamma):

    print("\n")
    print(f"Current Generation      : {generation}")
    print(f"Current Patches         : {curr_patches}")
    print(f"Current Training Loss   : {train_loss}")
    print(f"Current Validation Loss : {validation_loss}")
    print(f"Current Gamma Val       : {mean_gamma} ± {std_gamma}")
    print("\n")
    sys.stdout.flush()


def load_pretrained_model(model_path, device, lr):
    print(f"\nUsing pretrained Model: {model_path}\n")

    my_UNET = Dose3DUNET()
    state = torch.load(model_path, map_location=device)

    my_UNET.load_state_dict(state['model_state_dict'])

    optimizer = optim.Adam(my_UNET.parameters(),
                           lr, (0.9, 0.99), 10E-8)
    optimizer.load_state_dict(state['optimizer_state_dict'])

    if device.type == "cuda":
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

    return train_state


def no_pretrained_model(device, lr):

    my_UNET = Dose3DUNET()

    if device.type == "cuda":
        if torch.cuda.device_count() > 1:
            my_UNET = nn.DataParallel(my_UNET)
            print(f"Using {torch.cuda.device_count()} GPU's")

    optimizer = optim.Adam(my_UNET.parameters(),
                           lr, (0.9, 0.99), 10E-8)
    train_state = {
        'UNET': my_UNET,
        'optimizer': optimizer,
        'start_patch_num': 0,
        'epochs': [],
        'model_generation': 0
    }

    return train_state


def get_validation_segments(data_path, validation_segments):
    print("Validation Segments used: ", validation_segments)
    return [data_path + x for x in validation_segments]


def get_training_segments(data_path, validation_segments):
    return[data_path + x for x in os.listdir(data_path) if not x.startswith(".") and not data_path + x in validation_segments]


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

    device = utils.define_calculation_device(use_gpu)
    criterion = LOSS_FUNCTION

    if pretrained_model:
        train_state = load_pretrained_model(
            pretrained_model, device, learning_rate)

    else:
        train_state = no_pretrained_model(device, learning_rate)

    validation_segments = get_validation_segments(
        data_path, validation_segments)
    training_segments = get_training_segments(data_path, validation_segments)

    # hier segments_per_queue und patches_per_segment hochscruaben / runterschrauben
    train_queue = dataqueue_threading.DataQueue(
        segment_list=training_segments,
        batch_size=batch_size,
        segments_per_queue=SPQ_TRAIN,
        patch_size=patch_size,
        patches_per_segment=PPS_TRAIN,
        num_worker=8
    )

    validation_queue = dataqueue_threading.ValidationQueue(
        segments=validation_segments,
        batch_size=16
    )

    print(
        f"Training-Data shape: [{batch_size} ,5 , {patch_size}, {patch_size}, {patch_size}]\n")
    print(f"Learning Rate: {learning_rate}")

    sys.stdout.flush()

    training_params = {
        "batch_size": batch_size,
        "patch_size": patch_size,
        "lr": learning_rate
    }

    losses = train(
        train_state=train_state,
        num_patches=num_patches,
        train_queue=train_queue,
        validation_queue=validation_queue,
        save_dir=save_dir,
        criterion=criterion,
        device=device,
        training_parameter=training_params
    )

    fout = open(save_dir + "epochs_losses.pkl", "wb")
    pickle.dump(losses, fout)
    fout.close()


def tb_logger(logger, exp_name):

    if logger:
        NAME = logger
        print("Model information for Tensorboard saved under", NAME)
        logger = SummaryWriter(NAME)

    else:
        NAME = f"{LOGGER_DIR}model_{exp_name}_{time()}"
        print("Model information for Tensorboard saved under", NAME)
        logger = SummaryWriter(NAME)

    return logger


if __name__ == "__main__":

    args = parse()

    if args.root_dir[-1] != "/":
        args.root_dir += "/"

    DATA_DIRECTORY = args.root_dir

    LOGGER_SEGMENTS = [args.root_dir + LOGGER_SEGMENTS_LIST[0],
                       args.root_dir + LOGGER_SEGMENTS_LIST[1]]

    # 50 SEGMENTS PER QUEUE AND 20*BATCH_SIZE PATCHES PER SEGMENT
    # debug
    #SPQ_TRAIN = int(args.batch_size*2)
    #PPS_TRAIN = 50

    SPQ_TRAIN = int(args.batch_size*2)
    PPS_TRAIN = 500

    LOGGER = tb_logger(args.logger, args.experiment_name)

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

    LOGGER.close()
