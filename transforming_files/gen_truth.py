import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gen_stack_dict_19Jul import stacks

def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    # 返回旋转后的图像
    return rotated

def cut_resize_file(img_mat, cut_size=(512, 512)):
    # 根据最短边进行图像裁剪
    sp = img_mat.shape
    rows = sp[0]  # height(rows) of image
    cols = sp[1]  # width(colums) of image
    if rows >= cols:
        shorter = cols
    else:
        shorter = rows
    cropped = img_mat[0:shorter, 0:shorter]  # 裁剪坐标为[y0:y1, x0:x1]
    new_array = cv2.resize(cropped, cut_size, interpolation=cv2.INTER_NEAREST)
    return new_array

def gen_mask_list(mask_dir: str, rotation=None, flip=None):
    # print(mask_dir)
    mask_dirs = os.listdir(mask_dir)
    masks = []
    for f in mask_dirs:
        mask = cv2.imread(mask_dir + f)
        # print(mask)
        if mask is not None:
            # mask //= 127
            mask = np.uint8(cut_resize_file(mask))
            if rotation is not None:
                mask = rotate(mask, rotation)
            if flip is not None:
                mask = cv2.flip(mask, flip)
            new_mask = np.zeros(mask.shape, dtype=np.uint8)
            new_mask = np.uint8(mask > 127) + np.uint8(mask==127)*2
            masks.append(new_mask)
            # mask_chamber = np.uint8(mask==127)
            # mask_chamber = cut_resize_file(mask_chamber)
            # mask_myo = np.uint8(mask==255)
            # mask_myo = cut_resize_file(mask_myo)
            # if rotation is not None:
            #     mask_chamber = rotate(mask_chamber, rotation)
            #     mask_myo = rotate(mask_myo, rotation)
            # if flip is not None:
            #     mask_chamber = cv2.flip(mask_chamber, flip)
            #     mask_myo = cv2.flip(mask_myo, flip)
            # new_mask = 2 * mask_chamber + mask_myo
            # masks.append(new_mask)
            # if rotation is not None:
            #     mask = rotate(mask, rotation)
            # if flip is not None:
            #     mask = cv2.flip(mask, flip)
            # new_mask = np.zeros(mask.shape, dtype=np.uint8)
            # new_mask = np.uint8(mask > 127) + np.uint8(mask==127)*2
    return mask_dirs, masks

def write_masks(mask_dirs, masks, truth_mask_path, stack_name):
    for f, mask in zip(mask_dirs, masks):
        f_name = truth_mask_path + stack_name + '_' + f
        print(f_name)
        cv2.imwrite(f_name, mask)

def write_pics(start_slice, end_slice, start_t, end_t, orig_path, truth_pic_path, stack_name, rotation=None, flip=None):
    for s in range(start_slice, end_slice + 1):
        for t in range(start_t, end_t + 1):
            print(orig_path.format(s, t))
            pic = cv2.imread(orig_path.format(s, t))
            # if pic.shape != mask_shape:
            #    pic = pic[40:390, 40:476-30, ...]
            print(pic.shape)
            pic = cut_resize_file(pic)
            if rotation is not None:
                pic = rotate(pic, rotation)
            if flip is not None:
                pic = cv2.flip(pic, flip)
            # pic = rotate(pic, 270)
            # pic = cv2.flip(pic, 0)
            print(orig_path.format(s, t))
            cv2.imwrite(truth_pic_path \
                    + stack_name + f'_slice{s:03d}time{t:03d}.png', pic)

if __name__=='__main__':
    for idx, name in enumerate(stacks):
        # if name != '02MAR2021P1_1':
            # if idx == 10:
        stack_name = name
        #  print(stacks)
        paras_dir = stacks[name]["paras_dir"]
        mask_dir = paras_dir + "mask_with_chamber/"
        orig_path = paras_dir + "para{}_new/time{:03d}.png"
        start_slice = stacks[name]["start_slice"]
        end_slice = stacks[name]["end_slice"]
        start_t = stacks[name]["start_t"]
        end_t = stacks[name]["end_t"]
        rotation = stacks[name]["rotation"]
        flip = stacks[name]["flip"]
        truth_pic_path = '/home/haobo/Documents/UROP2021/Truth_pics_19Jul_healthy/all_set_f/'
        truth_mask_path = '/home/haobo/Documents/UROP2021/Truth_pics_19Jul_healthy/all_set_m/'
        mask_dirs, masks = gen_mask_list(mask_dir, rotation=rotation, flip=flip)
        mask_shape = masks[-1].shape
        write_masks(mask_dirs, masks, truth_mask_path, stack_name)
        # write_pics(start_slice, end_slice, start_t, end_t, orig_path, truth_pic_path, stack_name,rotation, flip)
        # cv2.imshow('', masks[-1]*100)
    # cv2.waitKey()