import numpy as np
import torch
from torch import nn
import astra
import matplotlib.pyplot as plt

def rescale(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x

def norm_maxmin(x, min=-1024.0, max=3072.0):
    x = (x - min) / (max - min)
    return x

class Operator(nn.Module):
    def __init__(self, pg, vg, I0=1e5):
        super(Operator, self).__init__()
        self.pg = pg
        self.vg = vg
        self.I0 = I0

        # # used for normalzation input
        # self.MAX = 0.032 / 5
        # self.MIN = 0

    # def fp(self, input):
    #     # vol and geo are used for the experimental setting
    #
    #     # ctx.save_for_backward(input) # only needed for non-linear operator
    #     # then at the backward function part , read the input
    #     # input, = ctx.saved_tensors
    #     # grad_input = None
    #
    #     out_shape = len(self.pg['ProjectionAngles']), self.pg['DetectorCount']  # shape of sinogram
    #     proj_id = astra.create_projector('cuda', self.pg, self.vg)
    #
    #     vid = astra.data2d.create('-vol', self.vg, 0)
    #     sid = astra.data2d.create('-sino', self.pg, 0)
    #     cfg = astra.astra_dict('FP_CUDA')
    #     cfg['ProjectorId'] = proj_id
    #     cfg['ProjectionDataId'] = sid
    #     cfg['VolumeDataId'] = vid
    #     fpid = astra.algorithm.create(cfg)
    #     # put value in the pointer
    #     input_arr = input.cpu().detach().numpy()
    #     # compute the extra_shape
    #     extra_shape = input_arr.shape[:-2]  # 2-D data
    #     if extra_shape:
    #         # Multiple inputs: flatten extra axes
    #         input_arr_flat_extra = input_arr.reshape((-1,) + input_arr.shape[-2:])
    #         results = []
    #         for inp in input_arr_flat_extra:
    #             astra.data2d.store(vid, inp)
    #             astra.algorithm.run(fpid)
    #             sino = astra.data2d.get(sid)
    #             # noisy_sino = astra.add_noise_to_sino(sino, self.I0)  # add poisson noise
    #             results.append(sino)
    #         # restore the correct shape
    #         result_arr = np.stack(results).astype(np.float32)
    #         result_arr = result_arr.reshape(extra_shape + out_shape)
    #     else:
    #         astra.data2d.store(vid, input_arr)
    #         astra.algorithm.run(fpid)
    #         sino = astra.data2d.get(sid)
    #         result_arr = sino
    #
    #     # convert back to tensor
    #     tensor = torch.from_numpy(result_arr).cuda()
    #     astra.projector.delete(proj_id)
    #     astra.algorithm.delete(fpid)
    #     astra.data2d.delete(vid)
    #     astra.data2d.delete(sid)
    #     return tensor

    def create_sino(self, input):
        out_shape = len(self.pg['ProjectionAngles']), self.pg['DetectorCount']  # shape of sinogram
        proj_id = astra.create_projector('cuda', self.pg, self.vg)

        # put value in the pointer
        input_arr = input.cpu().detach().numpy()
        extra_shape = input_arr.shape[:-2]  # 2-D data
        if extra_shape:
            # Multiple inputs: flatten extra axes
            input_arr_flat_extra = input_arr.reshape((-1,) + input_arr.shape[-2:])
            results = []
            for inp in input_arr_flat_extra:
                sino_id, sino = astra.create_sino(inp, proj_id)
                results.append(sino)
            # restore the correct shape
            result_arr = np.stack(results).astype(np.float32)
            result_arr = result_arr.reshape(extra_shape + out_shape)
        else:
            sino_id, sino = astra.create_sino(input_arr, proj_id)
            result_arr = sino

        # convert back to tensor
        tensor = torch.from_numpy(result_arr).cuda()
        astra.projector.delete(proj_id)
        astra.data2d.delete(sino_id)
        return tensor

    def fbp(self, input):

        out_shape = self.vg['GridRowCount'], self.vg['GridColCount']
        proj_id = astra.create_projector('cuda', self.pg, self.vg)

        vid = astra.data2d.create('-vol', self.vg, 0)
        sid = astra.data2d.create('-sino', self.pg, 0)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sid
        cfg['ReconstructionDataId'] = vid
        cfg['option'] = {'FilterType': 'Ram-Lak'}
        bpid = astra.algorithm.create(cfg)

        # Convert tensor to numpy

        input_arr = input.detach().cpu().numpy()
        extra_shape = input_arr.shape[:-2]  # 2-D data
        if extra_shape:
            # Multiple inputs: flatten extra axes
            input_arr_flat_extra = input_arr.reshape((-1,) + input_arr.shape[-2:])
            results = []
            for inp in input_arr_flat_extra:
                astra.data2d.store(sid, inp)
                astra.algorithm.run(bpid)
                results.append(astra.data2d.get(vid))
            # restore the correct shape
            result_arr = np.stack(results).astype(np.float32)
            result_arr = result_arr.reshape(extra_shape + out_shape)
        else:
            astra.data2d.store(sid, input_arr)
            astra.algorithm.run(bpid)
            result_arr = astra.data2d.get(vid)

        # convert back to tensor
        tensor = torch.from_numpy(result_arr).cuda()
        astra.projector.delete(proj_id)
        astra.algorithm.delete(bpid)
        astra.data2d.delete(vid)
        astra.data2d.delete(sid)
        return tensor

    def A(self, x):
        # m = self.I0 * torch.exp(-self.fp(x))
        # if add_noise:
        #     m = self.noise(m)

        sinogramRaw = self.create_sino(x)
        max_sinogramRaw = sinogramRaw.max()
        sinogramRawScaled = sinogramRaw / max_sinogramRaw
        # to detector count
        sinogramCT = self.I0 * torch.exp(-sinogramRawScaled)
        # add poison noise
        sinogramCT_C = torch.zeros_like(sinogramCT)
        for i in range(sinogramCT_C.shape[0]):
            for j in range(sinogramCT_C.shape[1]):
                sinogramCT_C[i, j] = torch.poisson(sinogramCT[i, j])
        # to density
        sinogramCT_D = sinogramCT_C / self.I0
        sinogram_out = -max_sinogramRaw * torch.log(sinogramCT_D)

        return sinogram_out


    # @staticmethod
    # def backward(ctx, grad_output):
    #     grad_input = None
    #
    #     if not ctx.needs_input_grad[2]:
    #         return None, None, None
    #
    #     proj_geo = ctx.proj_geo
    #     vol = ctx.vol
    #     in_shape = vol['GridRowCount'], vol['GridColCount']
    #     proj_id = astra.create_projector('cuda', proj_geo, vol)
    #
    #     vid = astra.data2d.create('-vol', vol, 0)
    #     sid = astra.data2d.create('-sino', proj_geo, 0)
    #     cfg = astra.astra_dict('BP_CUDA')
    #     cfg['ProjectorId'] = proj_id
    #     cfg['ProjectionDataId'] = sid
    #     cfg['ReconstructionDataId'] = vid
    #     bpid = astra.algorithm.create(cfg)
    #
    #     # Convert tensor to numpy
    #     grad_output_arr = grad_output.detach().cpu().numpy()
    #     extra_shape = grad_output_arr.shape[:-2]  # 2-D data
    #     if extra_shape:
    #         # Multiple inputs: flatten extra axes
    #         input_arr_flat_extra = grad_output_arr.reshape((-1,) + grad_output_arr.shape[-2:])
    #         results = []
    #         for inp in input_arr_flat_extra:
    #             astra.data2d.store(sid, inp)
    #             astra.algorithm.run(bpid)
    #             results.append(astra.data2d.get(vid))
    #         # restore the correct shape
    #         result_arr = np.stack(results).astype(np.float32)
    #         result_arr = result_arr.reshape(extra_shape + in_shape)
    #     else:
    #         astra.data2d.store(sid, grad_output_arr)
    #         astra.algorithm.run(bpid)
    #         result_arr = astra.data2d.get(vid)
    #     # convert back to tensor
    #     grad_input = torch.from_numpy(result_arr).cuda()
    #     astra.projector.delete(proj_id)
    #     astra.algorithm.delete(bpid)
    #     astra.data2d.delete(vid)
    #     astra.data2d.delete(sid)
    #     return None, None, grad_input



# if __name__ == '__main__':
#     img = np.load('data/test_data/L143_200_target.npy')
#     img = torch.from_numpy(img)
#     vg = astra.create_vol_geom(512, 512)
#     pg = astra.create_proj_geom('parallel', 1.0, 725, np.linspace(0, np.pi, 180, False))
#     I0 = 1e4
#     physics = Operator(pg, vg, I0)
#     # img = img * (physics.MAX - physics.MIN) + physics.MIN
#     sino = physics.create_sino(img)
#     rec = physics.fbp(sino)
#
#     print(img.shape, sino.shape, rec.shape)
#     print(img.min(), rec.min())
#
#     plt.figure(1)
#     plt.imshow(img.numpy().squeeze(), cmap='gray')
#     plt.figure(2)
#     plt.imshow(sino.cpu().numpy().squeeze(), cmap='gray')
#     plt.figure(3)
#     plt.imshow(rec.cpu().numpy().squeeze(), cmap='gray')
#     plt.show()
