import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from operatormodel import Operator, rescale
import astra
from dataset import DECT
from MD import DiffMBIR, dict2namespace
import logging
# from operatorct import OperatorModule, OperatorModuleT
from models.unet import UNet
import yaml
import skimage.metrics as skm
from utils.metrics import cal_psnr_ssim
import timeit


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # configs
    with open(os.path.join("configs", 'aapmct.yml'), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # path
    material = 'water'
    root_dir = 'data'
    save_dir = 'dect_material_decomposition'
    model_name = '{}_iterations'.format(material)
    save_path = os.path.join(save_dir, "results", model_name)
    os.makedirs(save_path, exist_ok=True)

    # CT operator
    n = config.data.image_size
    num_det = int(1.5*n)
    angles = config.data.sampling_angles
    vg = astra.create_vol_geom(n, n)
    pg = astra.create_proj_geom('parallel', 1.0, num_det, np.linspace(0, np.pi, angles, False))
    physics = Operator(pg, vg)
    physics = physics.to(device)

    # load data
    datasets = DECT(root_dir=root_dir, material=material, mode='test')
    dl = DataLoader(datasets, batch_size=1, shuffle=False)

    # load model
    sigma = 0.08
    md_model = DiffMBIR(config, CG=True, sigma=sigma)
    states = torch.load(os.path.join('dect_material_decomposition/logs/{}_initial_90_sino_unet'.format(material),
                                     "ckpt_40.pth"),
                            map_location=device)
    md_model.to(device)
    md_model.load_state_dict(states[0], strict=True)
    md_model.eval()

    for i, (y1, y2, tgt) in enumerate(dl):
        if i == 100:
            y1 = y1.to(device)
            y2 = y2.to(device)
            tgt = tgt.to(device)

            tic = timeit.default_timer()

            m, fbp = md_model(y1, y2)

            toc = timeit.default_timer()
            Run_time = toc - tic

            # plt.figure(1), plt.imshow(y1.detach().cpu().numpy().squeeze(), cmap='gray')
            # plt.figure(2), plt.imshow(y2.detach().cpu().numpy().squeeze(), cmap='gray')

            m = m.detach().cpu().numpy().squeeze()
            fbp = fbp.detach().cpu().numpy().squeeze()
            gt = tgt.detach().cpu().numpy().squeeze()

            break

    plt.figure(1), plt.imshow(m, cmap='gray')
    plt.figure(2), plt.imshow(fbp, cmap='gray')
    plt.figure(3), plt.imshow(gt, cmap='gray')
    plt.show()

    # data_range = gt.max() - gt.min()
    psnr_m, ssim_m = cal_psnr_ssim(m, gt)
    psnr_fbp, ssim_fbp = cal_psnr_ssim(fbp, gt)

    print(f"PSNR_diff:  {psnr_m:5.2f}")
    print(f"SSIM_diff:  {ssim_m:5.2f}")
    print(f"PSNR_fbp:  {psnr_fbp:5.2f}")
    print(f"SSIM_fbp:  {ssim_fbp:5.2f}")
    print("Diff-MBIR completed in {} seconds".format(Run_time))

    plt.imsave(os.path.join(save_path, 'gt.png'), gt, cmap='gray')
    plt.imsave(os.path.join(save_path, '{}_iterations.png'.format(config.sampling.iter_num)), m, cmap='gray')
    # plt.imsave(os.path.join(save_path, 'fbp.png'), fbp, cmap='gray')


if __name__ == '__main__':
    main()
