import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from operatormodel import Operator
import astra


class DECT(Dataset):
    """DECT dataset."""

    def __init__(self, root_dir, material='water', mode='train'):

        self.low = sorted(glob.glob(os.path.join(root_dir, 'CT_energysino', mode, 'low', '*.npy')))
        self.high = sorted(glob.glob(os.path.join(root_dir, 'CT_energysino', mode, 'high', '*.npy')))
        self.label = sorted(glob.glob(os.path.join(root_dir, material, mode, '*.npy')))

    def __len__(self):

        return len(self.low)

    def __getitem__(self, index):

        # get low- and high- energy sinogram
        y0 = np.load(self.low[index])
        y1 = np.load(self.high[index])

        # get target image
        tgt = np.load(self.label[index])
        tgt = (tgt - tgt.min()) / (tgt.max() - tgt.min())   # [0, 1]

        if y0.ndim == 2:
            y0 = y0[None, ...]
        y0 = torch.from_numpy(y0).to(torch.float32)

        if y1.ndim == 2:
            y1 = y1[None, ...]
        y1 = torch.from_numpy(y1).to(torch.float32)

        if tgt.ndim == 2:
            tgt = tgt[None, ...]
        tgt = torch.from_numpy(tgt).to(torch.float32)

        return y0, y1, tgt

if __name__ == '__main__':
    datasets = DECT(root_dir='D:\diffpir\DiffPIR\data', material='water', mode='train')
    print(len(datasets))
    dl = DataLoader(datasets, batch_size=1, shuffle=True)
    vg = astra.create_vol_geom(512, 512)
    pg = astra.create_proj_geom('parallel', 1.0, 768, np.linspace(0, np.pi, 90, False))
    physics = Operator(pg, vg)
    for (y0, y1, tgt) in dl:
        x0 = physics.fbp(y0)
        x1 = physics.fbp(y1)
        plt.figure(1), plt.imshow(y0.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.figure(2), plt.imshow(y1.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.figure(3), plt.imshow(x0.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.figure(4), plt.imshow(x1.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.figure(5), plt.imshow(tgt.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.show()

