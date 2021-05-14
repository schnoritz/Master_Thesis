import torch
import numpy as np
from time import time

path = "/Users/simongutwein/testfolder/"

tensor = torch.rand((512, 512, 110))
np_array = np.random.rand(512, 512, 110)

np.save(
    path + "np.npy",
    np_array,
)
torch.save(tensor, path + "torch.pt")

start = time()
np_array = np.load(path + "np.npy")
print(f"NUMPY TOOK {np.round(time()-start,2)} SECONDS")

start = time()
tensor = torch.load(path + "torch.pt")
print(f"TORCH TOOK {np.round(time()-start,2)} SECONDS")