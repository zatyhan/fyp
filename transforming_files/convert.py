#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Powered by haobo, 2021-09-29 10:35:31

import os
import numpy as np
import medImgProc.processFunc as pf
from cv2 import cv2
import SimpleITK as sitk

start_slice = 13
end_slice   = 26
anchor_1    = 7
anchor_2    = 30
case_path   = '../6. 03Sep2019P2/'
paras_path  = '../6. 03Sep2019P2/paras/'

for i in range(start_slice, end_slice + 1): 
    label_1 = cv2.imread(case_path + 'time{0:03d}/segmented/Segmented Slice{1:03d}.png'.format(anchor_1, i))
    print('time{0:03d}/segmented/Segmented Slice0{1:03d}.png'.format(anchor_1, i).format(anchor_1, i))
    label_2 = cv2.imread(case_path + 'time{0:03d}/segmented/Segmented Slice{1:03d}.png'.format(anchor_2, i))
    print(label_1.shape)
    label_1 = np.uint8(label_1 == 0)*100 + 50
    label_2 = np.uint8(label_2 == 0)*100 + 50

    cv2.imwrite(paras_path + 'para{}_new/label{}.png'.format(i, anchor_1), label_1)
    cv2.imwrite(paras_path + 'para{}_new/label{}.png'.format(i, anchor_2), label_2)

