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
        it += 1
    consts = np.asarray(consts, dtype=np.uint8)
    all_consts = np.prod(consts, axis=0).astype(np.uint8)
    # cv2.imshow('prod', 200*all_consts)
    # cv2.waitKey()
    # print(np.sum(all_consts))
    # print(all_consts.shape)
    print(img)
    all_consts *= img
    all_consts = np.uint8(all_consts != 0)
    # cv2.imshow('prod', 200*all_consts)
    # cv2.waitKey()
    return all_consts

def linear_interpolation(x, x0, x1, y0, y1):
    y = y0 * (1 - (x1 - x) / (x1 - x0)) + y1 * ((x1 - x) / (x1 - x0))
    return y

def load_imgs(start=1, end=0, dir=''):
    img_list = []
    for t in range(start, end + 1):
        img_path = dir + f'time{t:03d}.png'
        img = cv2.imread(img_path, 0)
        img_list.append(img)
        # print(img_path)
    return img_list

def remove_arrows(start_slice=1, end_slice=5, start_t=1, end_t=40, paras_path=''):
    for i in range(start_slice, end_slice + 1):
        path  = paras_path + 'para{}/'.format(i)
        path2 = paras_path + 'para{}_new/'.format(i)
        img_list = load_imgs(start=start_t, end=end_t, dir=path)
        all_consts = get_consts(img_list)
        if not os.path.exists(path2):
            os.makedirs(path2)

        dirs = os.listdir(path)
        for f in dirs:
            img = cv2.imread(paras_path + 'para{}/'.format(i) + f, 0)
            if img is not None and f[0] != 'S':
                listCoordmax0 = np.where(all_consts > 0)
                listCoordmax = list(zip(listCoordmax0[0], listCoordmax0[1]))
                listfilling = list(zip(listCoordmax0[0], listCoordmax0[1] - 5))
                img2 = img.copy()
                # print(listCoordmax0[1][1])
                # for cord in listCoordmax:
                #     print(cord)
                # for cord in listfilling:
                #     print(cord)
                for pos1, pos2 in zip(listCoordmax, listfilling):
                    img[pos1] = img2[pos2]
                # print(path+f)

                cv2.imwrite(path2 + f, img)

if __name__ == '__main__':
    paras_path = '/home/haobo/Documents/UROP2021/3. 14Jun2019 P1/paras/'
    remove_arrows(start_slice=1, end_slice=5, start_t=1, end_t=39, paras_path=paras_path)