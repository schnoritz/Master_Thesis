import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from model import Dose3DUNET
from time import time


def chunks(end, max_size):
    current = 0
    while current < end:
        chunk_size = min(max_size, end - current)
        yield (current + chunk_size)
        current += chunk_size


def predict_volume(inp, model, device, UQ=False):

    torch.cuda.empty_cache()

    num_devices = torch.cuda.device_count()
    slices = 16
    half_slice = slices // 2
    slices = [x for x in range(half_slice, inp.shape[-1]-half_slice+1, 10)]
    slices.append(inp.shape[-1]-half_slice)
    slices = list(dict.fromkeys(slices))

    prediction = np.zeros((2, *list([*inp.shape[1:]])))

    inputs = [inp[:, :, :, x-half_slice:x+half_slice] for x in slices]
    inputs = torch.stack(inputs)

    curr = 0
    print("Start Prediction")
    start = time()
    for chunk in chunks(len(inputs), num_devices):

        pred = predict(model, inputs[curr:chunk], device, UQ)
        for num, _slice in enumerate(slices[curr:chunk]):
            if num_devices == 1:
                prediction[0, :, :, _slice-half_slice:_slice+half_slice] = np.add(prediction[0, :, :, _slice-half_slice:_slice+half_slice], pred)
            else:
                prediction[0, :, :, _slice-half_slice:_slice+half_slice] = np.add(prediction[0, :, :, _slice-half_slice:_slice+half_slice], pred[num])

            prediction[1, :, :, _slice-half_slice:_slice+half_slice] += 1

        curr = chunk

    divisor = prediction[1]
    divisor[divisor == 0] = 1
    prediction[0] /= prediction[1]

    model.train()
    print("Prediction took: ", np.round(time()-start, 2), "seconds")
    torch.cuda.empty_cache()
    return torch.from_numpy(prediction[0])


def predict(model, inp, device, UQ):

    model.eval()
    with torch.no_grad():

        inp = inp.to(device)
        if UQ:
            pred, _ = model(inp)
            uncertainty = model.get_uncertainty(inp, 10)
            uncertainty = uncertainty.cpu().detach().numpy().squeeze()
        else:
            pred = model(inp)

        pred = pred.cpu().detach().numpy().squeeze()

        if UQ:
            return pred, uncertainty
        else:
            return pred


if __name__ == "__main__":

    input = torch.randn(5, 512, 512, 110)
    model = Dose3DUNET()

    predict_volume(inp=input, model=model, device=torch.device("cpu"))
