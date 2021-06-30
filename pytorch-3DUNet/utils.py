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

    return device


def save_model(model, optimizer, train_loss, test_loss, save_dir, epoch, save, epochs):

    if torch.cuda.device_count() > 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'epochs': epochs
        }, save_dir + f"UNET_epoch{epoch}.pt")
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'epochs': epochs
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


def get_training_data(train, target, device):

    train = train.float()
    target = target.float()

    if device.type == 'cuda':

        train = train.to(device)
        target = target.to(device)

    return train, target
