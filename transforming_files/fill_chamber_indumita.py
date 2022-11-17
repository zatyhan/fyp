import os
import numpy as np
import cv2

def draw_max_contour(img):
    mask = np.zeros(img.shape, dtype=np.uint8)
    cnts, _ = cv2.findContours(img, 
                     cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    c_max = cnts[0]
    # cv2.drawContours(mask, c_max, contourIdx=-1, color=255) #thickness=cv2.FILLED)
    cv2.fillPoly(mask, [c_max], 255)
    return c_max, mask

def draw_convex_hull(img, c_max):
    hull = cv2.convexHull(c_max)
    hull_filled = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(hull_filled, [hull], -1, 255, cv2.FILLED)
    return hull_filled

def fill_chamber(start_slice=1, end_slice=5, start_t=0, end_t=39, case_path='',
                 all_masked_path='', img_with_mask_path=''):
    # draw seg masks with chamber

    for t in range(start_t, end_t + 1):
        masked_path = case_path + f'time{t:03d}/mask_with_chamber/'   # destination folder of filled segmentation mask
        mkdir(img_with_mask_path)                                     # ensure all directories are valid
        mkdir(all_masked_path)
        mkdir(masked_path)

        for slice in range(start_slice, end_slice + 1):
            trf_path = case_path + f'time{t:03d}/Segmented/'
            img_path = trf_path + f'slice{slice:03d}time{t:03d}.png'
            print(img_path)
            img = cv2.imread(img_path, 0)
            
            # print(img.shape)
            c_max, contour_mask = draw_max_contour(img)
            # hull_filled = draw_convex_hull(img, c_max)
            # defects = hull_filled - contour_mask # area between contour and convex hull
            # _, mask = draw_max_contour(defects)
            mask = contour_mask                 # combined segmentation mask with chamber
            # orig_path = case_path + 'time{0:03d}/'.format(t) \
            #                       + 'slice{0:03d}time{1:03d}.png'.format(slice, t)
            orig_path = case_path + f"time{t:03d}/slice{slice:03d}time{t:03d}.png"    # the original path of images
            img_with_mask = cv2.imread(orig_path)
            # print(img_with_mask[..., 2].shape, mask.shape)
            # if img_with_mask[...,2].shape != mask.shape:        # a workaround for healthy heart 02Mar
            #     print(img_path)
            #     img_with_mask = img_with_mask[40:390, 40:476-30, ...]
            mask = 255 - mask
            img_with_mask[..., 2] = np.uint8(img_with_mask[..., 2] * 0.8 + mask * 0.2)       # mask on original image
            cv2.imwrite(all_masked_path + f'slice{slice:03d}time{t:03d}.png', mask)
            cv2.imwrite(masked_path + f'slice{slice:03d}time{t:03d}.png', mask)
            cv2.imwrite(img_with_mask_path \
                    + f'slice{slice:03d}time{t:03d}.png', img_with_mask)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    start_slice = 26
    end_slice   = 27
    start_t     = 0         # start_t and end_t start from 0 for consistence of registration
    end_t       = 19
    case_path = "/home/haobo/Documents/UROP2021/16. 08Oct2019P3/"
    all_masked_path = case_path + 'mask_with_chamber/'
    img_with_mask_path = case_path + 'img_with_mask/'

    fill_chamber(start_slice, end_slice, start_t, 
                end_t, case_path, all_masked_path, img_with_mask_path)

