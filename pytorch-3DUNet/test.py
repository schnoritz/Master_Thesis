import os
import torch
import pandas as pd
from model import Dose3DUNET
from time import time

model_path = "/mnt/qb/baumgartner/sgutwein84/save/bs32_ps32_5/UNET_896.pt"

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
model_checkpoint = torch.load(model_path, map_location=device)
model = Dose3DUNET(UQ=False)
model.load_state_dict(model_checkpoint['model_state_dict'])
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)

input = torch.randn((1, 5, 150, 150, 110))

start = time()
pred = model(input)
print(time()-start)

# model_dir = ["/mnt/qb/baumgartner/sgutwein84/save/bs32_ps32_5", "/mnt/qb/baumgartner/sgutwein84/save/bs32_ps32_4", "/mnt/qb/baumgartner/sgutwein84/save/bs32_ps32_6"]

# losses = []
# models = []
# for model_spec in model_dir:
#     for model in os.listdir(model_spec):

#         model_path = os.path.join(model_spec, model)
#         curr_model = torch.load(model_path, map_location="cpu")
#         models.append(model)
#         losses.append(curr_model['validation_loss'])

# data = pd.DataFrame(
#     {'model': models,
#      'loss': losses})

# data = data.sort_values(by=['loss'])
# print("Best Model :", data['model'].iloc[:5], data['loss'].iloc[:5])
