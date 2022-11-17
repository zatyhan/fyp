import os
import numpy as np
import cv2

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

pic_dir = "/home/haobo/Documents/UROP2021/HaoboSeg/val_f/"
label_dir = "/home/haobo/Documents/UROP2021/HaoboSeg/val_m/"
dst_dir = "/home/haobo/Documents/UROP2021/HaoboSeg/val_combined/"

mkdir(dst_dir)
pic_fns = os.listdir(pic_dir)
for pic_fn in pic_fns:
    pic = cv2.imread(pic_dir + pic_fn)
    label = cv2.imread(label_dir + pic_fn, 0)
    pic = pic.astype(np.uint8) * 0.8
    pic[..., 2] += label*51
    np.uint8(pic)
    cv2.imwrite(dst_dir + pic_fn, pic)