import skimage.metrics as skm
import cv2
import numpy as np



def cal_psnr_ssim(input, label):
    # input = (input - input.min()) / (input.max() - input.min())
    # label = (label - label.min()) / (label.max() - label.min())
    data_range = label.max() - label.min()
    psnr = skm.peak_signal_noise_ratio(label, input, data_range=data_range)
    ssim = skm.structural_similarity(label, input, data_range=data_range)

    return psnr, ssim

def read_img(path):
    # read image by cv2
    # return: Numpy float32, HW, grayscale, [0,1]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.

    return img

if __name__ == '__main__':
    material = 'bone'

    # gt = read_img('../dect_material_decomposition/results/water_90/gt_1.png')
    # fbp = read_img('../dect_material_decomposition/results/water_90/fbp_1.png')
    # diff = read_img('../dect_material_decomposition/results/water_90/diff_1.png')

    gt = read_img('../dect_material_decomposition/results/{}_90/gt_zoom.png'.format(material))
    fbp = read_img('../dect_material_decomposition/results/{}_90/fbp_zoom.png'.format(material))
    diff = read_img('../dect_material_decomposition/results/{}_90/diff_zoom.png'.format(material))
    F_TV = read_img('../dect_material_decomposition/results/{}_90/FISTA_TV_zoom.png'.format(material))
    N2C = read_img('../dect_material_decomposition/results/{}_90/N2C_zoom.png'.format(material))

    psnr_diff, ssim_diff = cal_psnr_ssim(diff, gt)
    psnr_fbp, ssim_fbp = cal_psnr_ssim(fbp, gt)
    psnr_FTV, ssim_FTV = cal_psnr_ssim(F_TV, gt)
    psnr_N2C, ssim_N2C = cal_psnr_ssim(N2C, gt)

    print(f"PSNR_diff:  {psnr_diff:5.2f}")
    print(f"SSIM_diff:  {ssim_diff:5.2f}")
    print(f"PSNR_fbp:  {psnr_fbp:5.2f}")
    print(f"SSIM_fbp:  {ssim_fbp:5.2f}")
    print(f"PSNR_FTV:  {psnr_FTV:5.2f}")
    print(f"SSIM_FTV:  {ssim_FTV:5.2f}")
    print(f"PSNR_N2C:  {psnr_N2C:5.2f}")
    print(f"SSIM_N2C:  {ssim_N2C:5.2f}")

    # gt = read_img('../dect_material_decomposition/results/bone_sigma/gt_zoom.png')
    # sigma = read_img('../dect_material_decomposition/results/bone_sigma/sigma0.1_zoom.png')
    #
    # psnr, ssim = cal_psnr_ssim(sigma, gt)
    # print(f"PSNR:  {psnr:5.2f}")
    # print(f"SSIM:  {ssim:5.2f}")
