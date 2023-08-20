import numpy as np
import matplotlib.pyplot as plt
import astra
import os
import glob
from tqdm import tqdm


def norm(input):
    newimg = (input - input.min()) / (input.max() - input.min())            # [0, 1]
    return newimg

def transform_ctdata(self, windowWidth, windowCenter, normal=True):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (self - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg

def bone_weighting(input, Ts, Tb):  # 1200, 1600
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j] > Tb:
                input[i, j] = 1 * input[i, j]
            elif input[i, j] < Ts:
                input[i, j] = 0 * input[i, j]
            else:
                input[i, j] = (input[i, j] - Ts) / (Tb - Ts) * input[i, j]
    return input

def water_weighting(input, Ts, Tb):
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j] > Tb:
                input[i, j] = 0 * input[i, j]
            elif input[i, j] < Ts:
                input[i, j] = 1 * input[i, j]
            else:
                input[i, j] = (Tb - input[i, j]) / (Tb - Ts) * input[i, j]
    return input

def main():
    mode = 'train'
    # data_path = 'D:/LIDC/LIDC_img/{}'.format(mode)
    data_path = 'data/CT_img/mono/{}'.format(mode)
    img_path = sorted(glob.glob(os.path.join(data_path, '*.npy')))
    bone_save_path = 'data/CT_img/bone/{}'.format(mode)
    os.makedirs(os.path.join(bone_save_path), exist_ok=True)
    water_save_path = 'data/CT_img/water/{}'.format(mode)
    os.makedirs(os.path.join(water_save_path), exist_ok=True)

    # continue_seg = sorted(glob.glob(os.path.join(water_save_path, '*.npy')))

    kvp_list = ['70', '120']   # options are "70" (low) and "120" (high)

    # save path
    low_sino_save_path = os.path.join('data/CT_energysino/{}/low'.format(mode))
    high_sino_save_path = os.path.join('data/CT_energysino/{}/high'.format(mode))
    low_transmission_path = os.path.join('data/CT_transmission/{}/low'.format(mode))
    high_transmission_path = os.path.join('data/CT_transmission/{}/high'.format(mode))
    os.makedirs(os.path.join(low_sino_save_path), exist_ok=True)
    os.makedirs(os.path.join(high_sino_save_path), exist_ok=True)
    os.makedirs(os.path.join(low_transmission_path), exist_ok=True)
    os.makedirs(os.path.join(high_transmission_path), exist_ok=True)

    for i in tqdm(range(len(img_path))):
        if i == 150:
            windowCenter = -120
            windowWidth = 1800
            img = np.load(img_path[i])
            img = transform_ctdata(img, windowWidth=windowWidth, windowCenter=windowCenter, normal=True)
            plt.figure(1), plt.imshow(img, cmap='gray')
            plt.show()

            Ts = 200
            Tb = 300
            minWindow = float(windowCenter) - 0.5 * float(windowWidth)
            new_Ts = (Ts - minWindow) / float(windowWidth)
            new_Tb = (Tb - minWindow) / float(windowWidth)

            img_bone = bone_weighting(np.array(img), Ts=new_Ts, Tb=new_Tb)
            img_water = water_weighting(np.array(img), Ts=new_Ts, Tb=new_Tb)

            # normalize to [0, 1]
            img_bone = norm(img_bone)
            img_water = norm(img_water)

            plt.figure(2), plt.imshow(img_bone, cmap='gray'), plt.axis('off')
            plt.figure(3), plt.imshow(img_water, cmap='gray'), plt.axis('off')
            plt.show()

            break

        windowCenter = -120
        windowWidth = 1800
        img = np.load(img_path[i])
        # img = img / 2000

        # adjust window width and level, and nomalize to [0, 1]
        img = transform_ctdata(img, windowWidth=windowWidth, windowCenter=windowCenter, normal=True)

        # plt.figure(1), plt.imshow(img, cmap='gray')
        # plt.show()
        # img = img + (0 - img.min())  # set the min value to 0

        # rescale threshold
        Ts = 200
        Tb = 300
        minWindow = float(windowCenter) - 0.5 * float(windowWidth)
        new_Ts = (Ts - minWindow) / float(windowWidth)
        new_Tb = (Tb - minWindow) / float(windowWidth)

        # segment
        img_bone = bone_weighting(np.array(img), Ts=new_Ts, Tb=new_Tb)
        img_water = water_weighting(np.array(img), Ts=new_Ts, Tb=new_Tb)

        # normalize to [0, 1]
        img_bone = norm(img_bone)
        img_water = norm(img_water)

        # # plt.figure(2), plt.imshow(img_bone, cmap='gray'), plt.axis('off')
        # # plt.figure(3), plt.imshow(img_water, cmap='gray'), plt.axis('off')
        # # plt.show()

        # save material images
        np.save(os.path.join(water_save_path, 'water_' + str(i)), img_water)
        np.save(os.path.join(bone_save_path, 'bone_' + str(i)), img_bone)

        # CT geometry
        n = 512
        num_det = int(1.5*n)
        angles = 180
        pg = astra.create_proj_geom('parallel', 1.0, num_det, np.linspace(0, np.pi, angles, False))
        vg = astra.create_vol_geom(n, n)
        proj_id = astra.create_projector('cuda', pg, vg)

        # material sinograms
        s_bone_id, s_bone = astra.create_sino(img_bone, proj_id)
        s_water_id, s_water = astra.create_sino(img_water, proj_id)

        astra.data2d.delete(s_water_id)
        astra.projector.delete(proj_id)

        # Create energy sinograms
        for k in range(2):
            kvp = kvp_list[k]
            # knock out every other view for kVp switching. First view is 70 kVp, second is 120 kVp, ...
            if kvp == "70":
                sbone = s_bone[1::2, :]
                swater = s_water[1::2, :]

            elif kvp == "120":
                sbone = s_bone[0::2, :]
                swater = s_water[0::2, :]

            modeldata = np.load(os.path.join('data', kvp + 'kvp_data.npy'))
            energies = modeldata[0] * 1.  # energy bins
            spectrum = modeldata[1] * 1.  # spectrum
            mu_bone = modeldata[2] * 1.  # bone mass attenuation coefficient
            mu_water = modeldata[3] * 1.  # water mass attenuation coefficient

            nviews, nbins = sbone.shape  # Grab the data dimensions from the Sinogram data

            delta_e = 0.5  # bin width

            transmission = np.zeros([nviews, nbins], "float32")
            for j in range(len(energies)):
                transmission += delta_e * spectrum[j] * np.exp(-mu_bone[j] * sbone - mu_water[j] * swater)

            # create energy sinograms
            energy_sino = -np.log(transmission)

            # UNCOMMENT if you want to save the data
            if kvp == "70":
                np.save(os.path.join(low_sino_save_path, str(kvp) + '_sino_' + str(i)), energy_sino.astype("float32"))
                # np.save(os.path.join(low_transmission_path, str(kvp) + '_trans_' + str(i)), transmission.astype("float32"))
            else:
                np.save(os.path.join(high_sino_save_path, str(kvp) + '_sino_' + str(i)), energy_sino.astype("float32"))
                # np.save(os.path.join(high_transmission_path, str(kvp) + '_trans_' + str(i)), transmission.astype("float32"))

if __name__ == '__main__':
    main()


























    # np.save(os.path.join(bone_save_path, 'bone_sino_' + str(i)), s_bone)
    # np.save(os.path.join(water_save_path, 'water_sino_' + str(i)), s_water)

    # plt.figure(1)
    # plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('(a) original image'), plt.axis('off')
    # plt.subplot(132), plt.imshow(img_bone, cmap='gray'), plt.title('(b) bone image'), plt.axis('off')
    # plt.subplot(133), plt.imshow(img_water, cmap='gray'), plt.title('(c) water image'), plt.axis('off')
    # plt.figure(2), plt.imshow(s_bone, cmap='gray')
    # plt.figure(3), plt.imshow(s_water, cmap='gray')
    # plt.show()


