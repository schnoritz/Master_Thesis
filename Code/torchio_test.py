import torchio as tio
import torch
from torch.utils.data import DataLoader
import numpy as np

if __name__ == "__main__":

    dat = np.load(
        "/Users/simongutwein/home/baumgartner/sgutwein84/container/output/p.npy"
    ).astype(np.float16)

    dat = torch.from_numpy(dat)
    torch.save(dat, "/Users/simongutwein/home/baumgartner/sgutwein84/container/output/p.pt")

    # patches = torch.randn((5,512,512,110))

    # sample_1 = tio.Subject(t1 = tio.ScalarImage(tensor=patches),
    #                        label = tio.LabelMap(tensor=patches>0.1))

    # sample_2 = tio.Subject(t1=tio.ScalarImage(tensor=patches),
    #                        label=tio.LabelMap(tensor=patches > 0.2))

    # sample_3 = tio.Subject(t1 = tio.ScalarImage(tensor=patches),
    #                        label = tio.LabelMap(tensor=patches>0.3))

    # print(sample_1.shape)

    # subjects_list = [sample_1, sample_2, sample_3]

    # spatial = tio.OneOf(
    #     {
    #         tio.RandomAffine(): 0.8,
    #         tio.RandomElasticDeformation(): 0.2,
    #     },
    #     p=0.75,
    # )

    # subjects_dataset = tio.SubjectsDataset(subjects_list, transform=spatial)
    # training_loader = DataLoader(subjects_dataset, batch_size=4, num_workers=4)

    # for subjects_batch in training_loader:
    #     inputs = subjects_batch['t1'][tio.DATA]
    #     target = subjects_batch['label'][tio.DATA]

    #     print(inputs.shape, target.shape)