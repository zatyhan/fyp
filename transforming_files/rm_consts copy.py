import os
import numpy as np
import cv2

def get_consts(img_list):
    it = 0
    consts = []
    for img in img_list:
        if it:
            
            const = img == img_list[0]
            consts.append(const)
            # print(const)
        it += 1
    consts = np.asarray(consts, dtype=np.uint8)
    print(np.sum(consts))
    all_consts = np.prod(consts, axis=0).astype(np.uint8)
    # cv2.imshow('prod', 200*all_consts)
    # cv2.waitKey()
    # print(np.sum(all_consts))
    # print(all_consts.shape)
    all_consts *= img
    all_consts = np.uint8(all_consts != 0)
    # cv2.imshow('prod', 200*all_consts)
    # cv2.waitKey()
    return all_consts

def load_imgs(start=1, end=0, t=1, dir=''):
    img_list = []
    for s in range(start, end + 1):
        img_path = dir + f'slice{s:03d}time{t:03d}.png'
        img = cv2.imread(img_path, 0)
        img_list.append(img)
        # print(img_list)
    return img_list

def remove_arrows():
    
    img_list = []
    path2 = '/home/haobo/Documents/UROP2021/HaoboSeg-onthefly-08Oct2019P1/test_f_new/'
    pic_path = '/home/haobo/Documents/UROP2021/HaoboSeg-onthefly-08Oct2019P1/test_f/'
    dirs = os.listdir(pic_path)
    # all_consts = get_consts(img_list)
    for f in dirs:
        img = cv2.imread(pic_path + f, 0)
        print(img)
        img_list.append(img)
    all_consts = get_consts(img_list)
    cv2.imshow('', all_consts*255)
    cv2.waitKey()
    for f in dirs:
        if img is not None and f[0] != 'S':
            listCoordmax0 = np.where(all_consts > 0)
            listCoordmax = list(zip(listCoordmax0[0], listCoordmax0[1]))
            listfilling = list(zip(listCoordmax0[0], listCoordmax0[1] - 5))

            # for cord in listCoordmax:
            #     print(cord)
            # for cord in listfilling:
            #     print(cord)
            for pos1, pos2 in zip(listCoordmax, listfilling):
                img[pos1] = img[pos2]
            # print(path+f)
            cv2.imwrite(path2 + f, img)

start_time = 1
end_time = 40
stack_path = '/home/haobo/Documents/UROP2021/3. 14Jun2019 P1/paras/'
remove_arrows()