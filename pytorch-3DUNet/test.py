import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from utils import save_model
from model import Dose3DUNET


if __name__ == "__main__":

    model = Dose3DUNET().float()
    optimizer = torch.optim.Adam(model.parameters())

    save_dir = "/Users/simongutwein/Studium/Masterarbeit/save/"

    save_model(
        model=model,
        optimizer=optimizer,
        train_loss=10,
        test_loss=15,
        save_dir=save_dir,
        epoch=100,
        save=True)

    model_info = torch.load(
        "/Users/simongutwein/Studium/Masterarbeit/save/UNET_epoch100.pt")

    model.load_state_dict(model_info['model_state_dict'])
    optimizer.load_state_dict(model_info['optimizer_state_dict'])
    curr_train_loss = model_info['train_loss']
    curr_test_loss = model_info['test_loss']
    epoch = model_info['epoch']

    model.eval()

    # do something here

    model.test()

    # start test routine here

    pass
