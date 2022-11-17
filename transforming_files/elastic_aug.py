import os

import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data.dataset import Dataset
import albumentations as albu

def elastic_transform(image_t, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
            Convolutional Neural Networks applied to Visual Document Analysis", in
            Proc. of the International Conference on Document Analysis and
            Recognition, 2003.

        Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image_t.shape
    shape_size = shape[:2]
    # print(image_t.shape)

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image_t, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def rand(self, a=0, b=1):
    return np.random.rand() * (b - a) + a



def get_random_data(image, label, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):

    # masks = [(label == v) for v in self.class_values]
    # mask = np.stack(masks, axis=-1).astype('float')

    if random:
        im_merge = np.concatenate((image[..., None], label[..., None]), axis=2)

        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                        im_merge.shape[1] * 0.08)
        image = im_merge_t[..., 0]
        mask = im_merge_t[..., 1]

    if mask.shape[-1] != 1:
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1).astype(mask.dtype)

    # print('image ', np.shape(image))
    #
    # image = np.transpose(preprocess_input(np.array(image, np.float64)), [2, 0, 1])
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(mask.dtype)

    # print('post ',np.shape(mask))

    return image, mask

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    ela_dir = '/home/haobo/HaoboSeg-pytorch/data_all/elastic_aug'
    dir_f = '/home/haobo/HaoboSeg-pytorch/data_all/elastic_aug/ela_f'
    dir_m = '/home/haobo/HaoboSeg-pytorch/data_all/elastic_aug/ela_m'
    dirs = os.listdir(dir_f)
    for iter in range(271, 300):
        print(f"this is iter {iter}")
        for path in dirs:
            mkdir(f'{ela_dir}/ela_f_{iter}')
            mkdir(f'{ela_dir}/ela_m_{iter}')
            # print(f'{dir_f}/{path}')
            image = cv2.imread(f'{dir_f}/{path}', 0)
            # print(image.shape)
            mask = cv2.imread(f'{dir_m}/{path}', 0)
            # print(mask.shape)
            image, mask = get_random_data(image, mask)
            cv2.imwrite(f'{ela_dir}/ela_f_{iter}/{path}', image)
            cv2.imwrite(f'{ela_dir}/ela_m_{iter}/{path}', mask)