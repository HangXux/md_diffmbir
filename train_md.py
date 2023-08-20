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
from MD import MD
import logging
import torch.utils.tensorboard as tb
from operatorct import OperatorModule, OperatorModuleT
from models.unet import UNet


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path
    material = 'bone'
    root_dir = 'D:/diffpir/DiffPIR/data'
    save_dir = 'D:/diffpir/DiffPIR/dect_material_decomposition'
    model_name = '{}_initial_90_sino_unet'.format(material)
    log_path = os.path.join(save_dir, "logs", model_name)
    os.makedirs(log_path, exist_ok=True)
    tb_path = os.path.join(save_dir, "tensorboard", model_name)
    os.makedirs(tb_path, exist_ok=True)

    # log
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    file_handler = logging.FileHandler(os.path.join(log_path, 'train_log.txt'))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format=log_format, level=logging.DEBUG, handlers=[file_handler, stream_handler])

    # CT operator
    n = 512
    num_det = int(1.5*n)
    angles = 90
    vg = astra.create_vol_geom(n, n)
    pg = astra.create_proj_geom('parallel', 1.0, num_det, np.linspace(0, np.pi, angles, False))
    physics = Operator(pg, vg)
    physics = physics.to(device)

    # load data
    datasets = DECT(root_dir=root_dir, material=material, mode='train')
    dl = DataLoader(datasets, batch_size=4, shuffle=True)

    # training config
    resume_training = True
    n_epochs = 100
    snapshot_freq = 20
    md_model = MD()
    md_model.to(device)
    optimizer = torch.optim.Adam(md_model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss().to(device)
    tb_logger = tb.SummaryWriter(log_dir=tb_path)

    start_epoch, step = 0, 0
    if resume_training:
        states = torch.load(os.path.join(log_path, "ckpt.pth"))
        md_model.load_state_dict(states[0])
        optimizer.load_state_dict(states[1])
        start_epoch = states[2]
        step = states[3]

    for epoch in range(start_epoch, n_epochs):
        for i, (y1, y2, tgt) in enumerate(dl):
            md_model.train()
            step += 1

            y1 = y1.to(device)
            y2 = y2.to(device)

            tgt = physics.create_sino(tgt)
            tgt = rescale(tgt)
            tgt = tgt.to(device)

            sino = md_model(y1, y2)

            # plt.figure(1), plt.imshow(y1[0].detach().cpu().numpy().squeeze(), cmap='gray')
            # plt.figure(2), plt.imshow(y2[0].detach().cpu().numpy().squeeze(), cmap='gray')
            # plt.figure(3), plt.imshow(sino[0].detach().cpu().numpy().squeeze(), cmap='gray')
            # plt.figure(4), plt.imshow(tgt[0].detach().cpu().numpy().squeeze(), cmap='gray')
            # plt.show()

            loss = criterion(sino, tgt)

            tb_logger.add_scalar("loss", loss, global_step=step)

            logging.info(
                f"epoch: {epoch + 1}, step: {step}, loss: {loss.item()}"
            )

            optimizer.zero_grad()

            # loss.requires_grad = True
            loss.backward()
            optimizer.step()

            if epoch % snapshot_freq == 0 or epoch == n_epochs-1:
                states = [
                    md_model.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                torch.save(
                    states,
                    os.path.join(log_path, "ckpt_{}.pth".format(epoch)),
                )
                torch.save(states, os.path.join(log_path, "ckpt.pth"))

if __name__ == "__main__":
    main()

