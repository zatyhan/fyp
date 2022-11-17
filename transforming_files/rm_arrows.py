#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Powered by haobo, 2021-09-30 12:20:06
import os
from cv2 import cv2
import numpy as np

start_slice = 13
end_slice   = 26
paras_path  = '../6. 03Sep2019P2/paras/'

for i in range(start_slice, end_slice + 1):
    path  = paras_path + 'para{}/'.format(i)
    path2 = paras_path + 'para{}_new/'.format(i)
    if not os.path.exists(path2):
        os.makedirs(path2)

    dirs = os.listdir(path)
    for f in dirs:
        img = cv2.imread(paras_path + 'para{}/'.format(i) + f, 0)
        if img is not None and f[0] != 'S':
            listCoordmax0 = np.where(img > 200)
            listCoordmax = list(zip(listCoordmax0[0], listCoordmax0[1]))
            listfilling = list(zip(listCoordmax0[0], listCoordmax0[1] - 5))

            # for cord in listCoordmax:
            #     print(cord)
            # for cord in listfilling:
            #     print(cord)
            for pos1, pos2 in zip(listCoordmax, listfilling):
                img[pos1] = img[pos2]
            print(path+f)
            cv2.imwrite(path2 + f, img)
