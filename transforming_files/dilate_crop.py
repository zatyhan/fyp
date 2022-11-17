import os
import cv2
import numpy as np

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def mksquare(src):
    max_dim = np.max(src.shape)
    min_dim = np.min(src.shape)
    dst = np.zeros((max_dim, max_dim), np.uint8)
    if max_dim == src.shape[0]:
        left = max_dim//2 - min_dim//2
        right = left + min_dim
        dst[:, left:right] = src
    elif max_dim == src.shape[1]:
        top = max_dim//2 - min_dim//2
        bottom = top + min_dim
        dst[top:bottom, :] = src
    dst = cv2.resize(dst, (256, 256), interpolation=cv2.INTER_CUBIC)
    return dst

def dilate_crop():
    pic = None
    mask = None
    pic_dir = '/home/haobo/HaoboSeg-pytorch/data_all/elastic_aug/ela_f_{}/'
    mask_dir = '/home/haobo/HaoboSeg-pytorch/data_all/elastic_aug/ela_m_{}/'
    pic_cropped_dir = '/home/haobo/HaoboSeg-pytorch/data_all/elastic_aug/ela_f_cropped_{}/'
    mask_cropped_dir = '/home/haobo/HaoboSeg-pytorch/data_all/elastic_aug/ela_m_cropped_{}/'
    # masked_cropped_dir = '/home/haobo/HaoboSeg-pytorch/data_all/masked_cropped/'
    # pic_dir = '/home/haobo/Documents/UROP2021/HaoboSeg-onthefly-08Oct2019P1/test_f_new/'
    # mask_dir = '/home/haobo/Documents/UROP2021/HaoboSeg-onthefly-08Oct2019P1/test_m_new/'
    # pic_cropped_dir = '/home/haobo/Documents/UROP2021/HaoboSeg-onthefly-08Oct2019P1/test_f_cropped_new/'
    # mask_cropped_dir = '/home/haobo/Documents/UROP2021/HaoboSeg-onthefly-08Oct2019P1/test_m_cropped_new/'
    for iter in range(300):
        mkdir(pic_cropped_dir.format(iter))
        mkdir(mask_cropped_dir.format(iter))
        # mkdir(masked_cropped_dir)
        pics_path = os.listdir(pic_dir.format(iter))
        kernel = np.ones((20,20), np.uint8) 
        for pic_name in pics_path:
            pic_path = os.path.join(pic_dir.format(iter), pic_name)
            mask_path = os.path.join(mask_dir.format(iter), pic_name)
            print(mask_path)
            pic = cv2.imread(pic_path, 0)
            mask = cv2.imread(mask_path, 0)
            print(mask.shape)
            mask_dilated = cv2.dilate(mask, kernel)
            mask_dilated = np.uint8(mask_dilated > 0)*255
            cnts, _ = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            cnt = cnts[0]
            x,y,w,h = cv2.boundingRect(cnt)
            # print(x, y, w, h)
            mask_cropped = mask[y:y + h, x:x + w]
            pic_cropped = pic[y:y + h, x:x + w]
            # print(mask_cropped.shape)
            mask_cropped = mksquare(mask_cropped)
            pic_cropped = mksquare(pic_cropped)
            cv2.imwrite(pic_cropped_dir.format(iter) + pic_name, pic_cropped)
            cv2.imwrite(mask_cropped_dir.format(iter) + pic_name, mask_cropped)
            # masked_cropped_2= mask_cropped*50 + pic_cropped*0.8
            # masked_cropped = np.zeros((*pic_cropped.shape, 3))
            # masked_cropped[..., 0] = pic_cropped
            # masked_cropped[..., 1] = pic_cropped
            # print(masked_cropped.shape)
            # masked_cropped[..., 2] = np.uint8(pic_cropped* 0.8 + mask_cropped *25)   
            # cv2.imwrite(masked_cropped_dir + pic_name, masked_cropped)
        # masked_pic = np.uint8(mask_cropped*50 + pic_cropped*0.8)
        # cv2.imshow('masked', masked_pic)
        # cv2.waitKey()

dilate_crop()