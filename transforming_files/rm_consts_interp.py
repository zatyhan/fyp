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
    # print(img)
    all_consts *= img
    all_consts = np.uint8(all_consts != 0)
    # cv2.imshow('prod', 200*all_consts)
    # cv2.waitKey()
    return all_consts

def linear_interpolation(x, x0, x1, y0, y1):
    """
    y0: The intensity of the pixel to the left of the defect in a given line index(e.g. an arrow)
    y1: The internsity of the pixel to the right of the defect in a given line index
    x0: The horizontal position of the pixel to the left of the defect given line idx
    x1: The horizontal position of the pixel to the right of the defect given line idx
    x:  The horizontal position of the pixel to be interpolated

    Returns: 
    y: the linearly interpolated intersity
    """
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

def remove_arrows(start_slice=1, end_slice=63, start_t=1, end_t=37, paras_path=''):
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
            print(paras_path + 'para{}/'.format(i) + f)
            if img is not None and f[0] != 'S':
                listCoordmax0 = np.where(all_consts > 0)
                # listCoordmax = list(zip(listCoordmax0[0], listCoordmax0[1]))
                # listfilling = list(zip(listCoordmax0[0], listCoordmax0[1] - 5))
                img2 = img.copy()
                listCoordmax0 = np.asarray(listCoordmax0)
                listCoordmax0 = np.transpose(listCoordmax0)
                unique_coord0 = np.unique(listCoordmax0[..., 0])
                # print(unique_coord0)
                for coord0 in unique_coord0:
                    positions = np.where(listCoordmax0[..., 0] == coord0)
                    for pos in positions:
                        interp_line = listCoordmax0[pos]
                        line_idx = interp_line[0, 0]
                        x0 = interp_line[..., 1].min() - 1
                        x1 = interp_line[..., 1].max() + 1
                        y0 = img[line_idx, x0]
                        y1 = img[line_idx, x1]
                        # print(y0, y1)
                        # print((line_idx, x1))
                        for point in interp_line:
                            x = point[1]
                            y = np.round(linear_interpolation(x, x0, x1, y0, y1)).astype(int)
                            # print(y)
                            idx = (point[0], point[1])
                            # print(img2[idx])
                            img2[idx] = y

                # print(x, x0, x1, y0, y1)
                cv2.imwrite(path2 + f, img2)

if __name__ == '__main__':
    paras_path = '/Users/nureizzatyhamzaid/Library/Mobile Documents/com~apple~CloudDocs/FYP_23/Disease/Case 99/Pre_Slices/paras/'
    remove_arrows(start_slice=29, end_slice=41, start_t=1, end_t=40, paras_path=paras_path)