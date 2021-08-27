import os
import torch
import pandas as pd
from model import Dose3DUNET
from time import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob

# masks = torch.load("/mnt/qb/baumgartner/sgutwein84/training_prostate/p11_47/training_data.pt")

# for num, mask in enumerate(masks):

#     plt.imshow(mask[:, :, 40], cmap="plasma")
#     plt.axis("off")
#     plt.savefig(f"/mnt/qb/baumgartner/sgutwein84/test/masks_{num}.png")

save_dir = "/mnt/qb/baumgartner/sgutwein84/save/"
model_dir = [os.path.join(save_dir, x) for x in glob.glob(save_dir + "*") if not "UQ" in x]
model_dir = ["/mnt/qb/baumgartner/sgutwein84/save/bs128_ps32_lr4_2108231722"]
print(model_dir)

pd.set_option('display.max_colwidth', None)

losses = []
models = []
for model_spec in model_dir:
    for model in os.listdir(model_spec):

        model_path = os.path.join(model_spec, model)
        curr_model = torch.load(model_path, map_location="cpu")
        models.append(model_path)
        losses.append(curr_model['validation_loss'])

data = pd.DataFrame(
    {'model': models,
     'loss': losses})

data = data.sort_values(by=['loss'])
print("Best Model :\n", data['model'].iloc[:5], "\n", data['loss'].iloc[:5])
