import torch
import torch.nn as nn
import numpy as np
import os
from guided_diffusion.script_util import create_gaussian_diffusion
import argparse
from models.diffusion import Model
import astra
from operatormodel import Operator, rescale
from utils import utils_model
import yaml
import matplotlib.pyplot as plt
from models.ema import EMAHelper
from models.unet import UNet
from operatorct import OperatorModule, OperatorModuleT


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

# Diff-MBIR algorithm ======================
def diffpir(sino, config, CG=True, sigma=0.08):
    num_diffusion_timesteps = config.diffusion.num_diffusion_timesteps
    iter_num = config.sampling.iter_num  # set number of sampling iterations
    iter_num_U = config.sampling.iter_num_U  # set number of inner iterations, default: 1
    sigma = sigma  # noise level associated with condition y
    lambda_ = config.sampling.lambda_  # key parameter lambda
    ddim_sample = False  # sampling method
    model_output_type = config.sampling.model_output_type  # model output type: pred_x_prev; pred_xstart; epsilon; score
    generate_mode = 'DiffPIR'
    skip_type = config.sampling.skip_type  # uniform, quad
    eta = 0.  # eta for ddim sampling
    zeta = config.sampling.zeta
    guidance_scale = 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # noise schedule
    beta_start = config.diffusion.beta_start
    beta_end = config.diffusion.beta_end
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)  # equivalent noise sigma on image
    noise_model_t = 0
    t_start = num_diffusion_timesteps - 1

    # ----------------------------------------
    # load pretrained DDPM model
    # ----------------------------------------

    diffusion = create_gaussian_diffusion()
    model = Model(config)
    states = torch.load(os.path.join('dect_material_decomposition/logs/{}_diffusion'.format(config.data.category),
                                     "ckpt_1100000.pth"),
                        map_location=device)
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(states[0], strict=True)
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(model)

    model.eval()

    if generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False

    # --------------------------------
    # load operator
    # --------------------------------

    n = config.data.image_size
    det_num = int(n*1.5)
    angles = config.data.sampling_angles
    vg = astra.create_vol_geom(n, n)
    pg = astra.create_proj_geom('parallel', 1.0, det_num, np.linspace(0, np.pi, angles, False))
    physics = Operator(pg, vg)

    # --------------------------------
    # (1) get img_L
    # --------------------------------

    # sino = physics.create_sino(sino)
    rec = physics.fbp(sino)  # reconstruct sparse view image
    rec = rescale(rec)  # [0, 1]
    # plt.figure(1), plt.imshow(rec.detach().cpu().numpy().squeeze(), cmap='gray')

    # --------------------------------
    # (2) get rhos and sigmas
    # --------------------------------

    sigmas = []
    sigma_ks = []
    rhos = []
    for i in range(num_diffusion_timesteps):
        sigmas.append(reduced_alpha_cumprod[num_diffusion_timesteps - 1 - i])
        if model_output_type == 'pred_xstart' and generate_mode == 'DiffPIR':
            sigma_ks.append((sqrt_1m_alphas_cumprod[i] / sqrt_alphas_cumprod[i]))
        else:
            sigma_ks.append(torch.sqrt(betas[i] / alphas[i]))
        rhos.append(lambda_ * (sigma ** 2) / (sigma_ks[i] ** 2))

    rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), \
                             torch.tensor(sigma_ks).to(device)

    # --------------------------------
    # (3) initialize x
    # --------------------------------

    x_t = sqrt_alphas_cumprod[t_start] * (2 * rec - 1) + sqrt_1m_alphas_cumprod[t_start] * torch.randn_like(rec)
    # rescale [-1, 1]

    # --------------------------------
    # (4) main iterations
    # --------------------------------

    # create sequence of timestep for sampling
    skip = num_diffusion_timesteps // iter_num
    if skip_type == 'uniform':
        seq = [i * skip for i in range(iter_num)]
        if skip > 1:
            seq.append(num_diffusion_timesteps - 1)
    elif skip_type == "quad":
        seq = np.sqrt(np.linspace(0, num_diffusion_timesteps ** 2, iter_num))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1

    # reverse diffusion for one image from random noise
    for i in range(len(seq)):
        curr_sigma = sigmas[seq[i]].cpu().numpy()
        # time step associated with the noise level sigmas[i]
        t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
        # skip iters
        if t_i > t_start:
            continue
        # repeat for semantic consistence: from repaint
        for u in range(iter_num_U):
            # generate denoised output z from diffusion model
            if generate_mode == 'DiffPIR':
                z = utils_model.model_fn(x_t, noise_level=curr_sigma * 255, model_out_type=model_output_type,
                                         model_diffusion=model, diffusion=diffusion,
                                         ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
            else:
                z = utils_model.model_fn(x_t, noise_level=curr_sigma * 255, model_out_type='pred_x_prev',
                                         model_diffusion=model, diffusion=diffusion,
                                         ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

            # --------------------------------
            # step 2, conjugate gradient
            # --------------------------------

            if seq[i] != seq[-1]:
                if generate_mode == 'DiffPIR':
                    if CG:
                        if model_output_type == 'pred_xstart':
                            # when noise level less than given image noise, skip
                            if i < num_diffusion_timesteps - noise_model_t:
                                rho = rhos[t_i].float().repeat(1, 1, 1, 1)
                                # z = z / 2 + 0.5  # [0, 1]
                                z = rescale(z)
                                dc = data_consistency(rho, physics)  # data consistency function
                                x0 = dc(rec, z)
                                x0 = x0 * 2 - 1  # [-1, 1]
                                # x0 = z + guidance_scale * (x0 - z)

                            else:
                                model_output_type = 'pred_x_prev'
                                x0 = utils_model.model_fn(x_t, noise_level=curr_sigma * 255,
                                                          model_out_type=model_output_type,
                                                          model_diffusion=model, diffusion=diffusion,
                                                          ddim_sample=ddim_sample,
                                                          alphas_cumprod=alphas_cumprod)

                                pass

                    else:
                        z = z.requires_grad_()
                        # first order solver
                        A = OperatorModule(pg, vg)
                        AT = OperatorModuleT(pg, vg)
                        meas = A(z)
                        ATA = AT(meas)
                        norm_grad, norm = utils_model.grad_and_value(operator=ATA, x=z, x_hat=z,
                                                                     measurement=rec)

                        x0 = z - norm_grad * norm / (rhos[t_i])
                        x0 = x0.detach_()
                        pass

                else:
                    x0 = z

            # add noise back to t=i-1
            if (generate_mode == 'DiffPIR' and model_output_type == 'pred_xstart') and not \
                    (seq[i] == seq[-1] and u == iter_num_U - 1):

                t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,
                                                 sigmas[seq[i + 1]].cpu().numpy())
                eps = (x_t - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                # calculate \hat{\eposilon}
                eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[
                    t_i] * torch.sqrt(betas[t_i])
                x_t = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1 - zeta) * (
                        torch.sqrt(sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma ** 2) * eps
                        + eta_sigma * torch.randn_like(x_t)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[
                          t_im1] * torch.randn_like(x_t)
            else:
                x_t = x0
                pass

            # set back to x_t from x_{t-1}
            if u < iter_num_U - 1 and seq[i] != seq[-1]:
                ### it's equivalent to use x & xt (?), but with xt the computation is faster.
                # x = torch.sqrt(alphas[t_i]) * x + torch.sqrt(betas[t_i]) * torch.randn_like(x)
                sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                x_t = sqrt_alpha_effective * x_t + torch.sqrt(sqrt_1m_alphas_cumprod[t_i] ** 2 -
                                                              sqrt_alpha_effective ** 2 *
                                                              sqrt_1m_alphas_cumprod[t_im1] ** 2) * torch.randn_like(x_t)

        x_0 = (x_t / 2 + 0.5)  # [0, 1]
        # x_0 = rescale(x_t)

    return x_0


# CG algorithm ======================
class myAtA(nn.Module):
    def __init__(self, rho, physics):
        super(myAtA, self).__init__()
        self.rho = rho
        self.physics = physics

    def forward(self, im):  # step for batch image
        sino = self.physics.create_sino(im)
        rec = self.physics.fbp(sino)

        return rec + self.rho * im


def myCG(AtA, rhs):
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(r * r)

    while i < 10 and rTr > 1e-10:
        Ap = AtA(p)
        aaa = torch.sum(p * Ap)
        alpha = rTr / aaa
        alpha = alpha
        x = x + alpha * p
        r = r - alpha * Ap
        bbb = torch.sum(r * r)
        rTrNew = bbb
        beta = rTrNew / rTr
        beta = beta
        p = r + beta * p
        i += 1
        rTr = rTrNew

    return x


class data_consistency(nn.Module):
    def __init__(self, rho, physics):
        super().__init__()
        self.rho = rho
        self.physics = physics

    def forward(self, rec, z):
        rhs = rec + self.rho * z
        AtA = myAtA(self.rho, self.physics)
        rec = myCG(AtA, rhs)

        return rec


class DiffMBIR(nn.Module):
    def __init__(self, config, CG=True, sigma=0.08):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.config = config
        self.unet = UNet(in_channels=2, out_channels=1, compact=3, residual=False)
        self.CG = CG
        self.sigma = sigma

    def forward(self, y1, y2):
        """
        :x0: zero-filled reconstruction atb
        """

        y = torch.cat([y1, y2], dim=1)
        sino = self.unet(y)
        sino = rescale(sino)
        m = diffpir(sino, self.config, CG=self.CG, sigma=self.sigma)

        return m

class MD(nn.Module):
    def __init__(self, physics):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.physics = physics
        self.unet = UNet(in_channels=2, out_channels=1, compact=3, residual=False)


    def forward(self, y1, y2):
        """
        :x0: zero-filled reconstruction atb
        """

        y = torch.cat([y1, y2], dim=1)
        sino = self.unet(y)

        # m = self.physics(sino)

        return rescale(sino)

# if __name__ == '__main__':
#     with open(os.path.join("configs", 'aapmct.yml'), "r") as f:
#         config = yaml.safe_load(f)
#     config = dict2namespace(config)
#
#     # path = os.path.join('testsets', 'ct')
#     # paths = util.get_image_paths(path)
#     img = np.load('testsets/ct/water_seg_60.npy')
#     img = rescale(img)
#     img = img[None, None, ...]
#     img = torch.from_numpy(img).to(torch.float32)
#     img = img.to(torch.device('cuda'))
#     rec = diffpir(img, config)
#
#     plt.figure(1), plt.imshow(img.detach().cpu().squeeze().numpy(), cmap='gray')
#     plt.figure(2), plt.imshow(rec.detach().cpu().squeeze().numpy(), cmap='gray')
#     plt.show()
