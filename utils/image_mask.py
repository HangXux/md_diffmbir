import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def get_mask_image(mask, left_top, right_top, left_bottom, right_bottom):
    # 显示anchor的图像 顺序必须为左上，左下，右下，右上
    contours = np.array([[left_top, left_bottom, right_bottom, right_top]], dtype=np.int32)
    # contours = np.array([[left, top], [left, bottom], [right, bottom], [right, top]], dtype=np.int32)
    # print(contours)
    """
    第一个参数是显示在哪个图像上；
    第二个参数是轮廓；
    第三个参数是指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓；
    第四个参数是绘制的颜色；
    第五个参数是线条的粗细
    """
    mask_image = cv2.drawContours(mask, contours, -1, (0, 0, 255), 1)  # 颜色：BGR
    # cv2.imshow('drawimg', mask_image)
    # cv2.waitKey(0)
    return mask_image

if __name__ == '__main__':
    material = 'bone'
    filename = 'N2C'
    image_path = "../dect_material_decomposition/results/{}_90".format(material)
    original_image = cv2.imread(os.path.join(image_path, '{}.png'.format(filename)))
    # print(original_image.shape)
    original_image_width = original_image.shape[1]
    original_image_height = original_image.shape[0]
    print("该图像尺寸(宽*高)为：{}*{}".format(original_image_width, original_image_height))

    # bone image
    if material == 'bone':
        left_top = [320, 180] # anchor左上角的坐标
        right_top = [420, 180] # anchor右上角的坐标
        left_bottom = [320, 280] # anchor左下角的坐标
        right_bottom = [420, 280] # anchor右下角的坐标
        # left, right, top, bottom = 200, 300, 310, 410
        # left_top = [left, top]  # anchor左上角的坐标
        # right_top = [right, top]  # anchor右上角的坐标
        # left_bottom = [left, bottom]  # anchor左下角的坐标
        # right_bottom = [right, bottom]  # anchor右下角的坐标

    # soft iamge
    if material == 'water':
        left_top = [270, 100]  # anchor左上角的坐标
        right_top = [370, 100]  # anchor右上角的坐标
        left_bottom = [270, 200]  # anchor左下角的坐标
        right_bottom = [370, 200]  # anchor右下角的坐标
        # left, right, top, bottom = 270, 370, 70, 170
        # left_top = [left, top]  # anchor左上角的坐标
        # right_top = [right, top]  # anchor右上角的坐标
        # left_bottom = [left, bottom]  # anchor左下角的坐标
        # right_bottom = [right, bottom]  # anchor右下角的坐标

    mask = original_image.copy()
    mask_image = get_mask_image(mask, left_top, right_top, left_bottom, right_bottom)

    x1 = min(left_top[0], right_top[0], left_bottom[0], right_bottom[0])
    x2 = max(left_top[0], right_top[0], left_bottom[0], right_bottom[0])
    y1 = min(left_top[1], right_top[1], left_bottom[1], right_bottom[1])
    y2 = max(left_top[1], right_top[1], left_bottom[1], right_bottom[1])
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_image[y1:y1 + hight, x1:x1 + width] # 得到剪切后的图像
    # print(crop_img.shape)
    # cv2.imshow('cuttimg', crop_img)
    # cv2.waitKey(0)

    img = Image.fromarray(crop_img)
    # 这里如果没有mask直接操作原图，那么剪切后的图像会带个蓝框
    # 因为上边生成mask_image的时候颜色顺序是BGR，但是这里是RGB
    # img.show()

    img = img.resize((original_image_width, original_image_height))
    # img.show()

    # 给放大的图加红色框
    # left_top = [0, 0]  # anchor左上角的坐标
    # right_top = [original_image_width, 0]  # anchor右上角的坐标
    # left_bottom = [0, original_image_height]  # anchor左下角的坐标
    # right_bottom = [original_image_width, original_image_height]  # anchor右下角的坐标
    img = np.array(img)
    # mask_crop_img = get_mask_image(img, left_top, right_top, left_bottom, right_bottom)

    result_img = np.vstack((mask_image, img))
    cv2.imwrite(os.path.join(image_path, '{}_stack.png'.format(filename)), result_img)
    cv2.imwrite(os.path.join(image_path, '{}_zoom.png'.format(filename)), img)

    plt.imshow(img)
    plt.show()
