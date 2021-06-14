import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os


def plot_patches(patches, target, idx):

    fig, axs = plt.subplots(1, 6, figsize=(24, 4))
    img = 0
    for b in patches:
        i = 0
        img += 1
        for p in b:
            axs[i].imshow(p[:, :, p.shape[2]//2])
            i += 1

        axs[5].imshow(target[img-1, 0, :, :, target.shape[2]//2])
        axs[5].set_title("Target Patch")
        for i in range(5):
            axs[i].set_title("Training Patch")

        name = "/home/baumgartner/sgutwein84/training_data/output_images/image" + \
            str(img) + str(idx) + ".png"

        if os.path.isfile(name):
            os.remove(name)

        plt.savefig(name)

    plt.close()


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


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


def save_model(model, optimizer, train_loss, test_loss, save_dir, epoch, save):

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, save_dir + f"UNET_epoch{epoch}.pt")

    if type(save) == int:
        os.remove(save_dir + f"UNET_epoch{save}.pt")


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
