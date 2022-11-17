#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Powered by haobo, 2021-09-29 10:35:31

import os
import numpy as np
import medImgProc.processFunc as pf
from cv2 import cv2
import SimpleITK as sitk
import numpy as np

start_slice = 13
end_slice   = 26
start_t     = 0
end_t       = 39
label_t     = 6
paras_path  = '../6. 03Sep2019P2/paras/'
label_file  = 'label7.png'

for slice in range(start_slice, end_slice + 1):
    label = cv2.imread(paras_path + '/para{}_new/'.format(slice) + label_file,  0)
    stlFile = cv2.GaussianBlur(label, (5,5),20)
    print(label)
    for t in range(start_t, end_t + 1):
        if t != label_t:
            trfFile = paras_path + '/para{}_new/bsfTransform/t{}to{}.txt'.format(slice, t, label_t)
            # print(trfFile)
            pf.transform_img2img(stlFile, trfFile,savePath=paras_path + 'para{}_new'.format(slice),fileName='trf{}'.format(t), scale=0.196078)
            trf_dir = paras_path + '/para{}_new/trfs/'.format(slice)
            if not os.path.exists(trf_dir):
                os.makedirs(trf_dir)
            img = sitk.ReadImage(paras_path + '/para{}_new/trf{}.mha'.format(slice, t))
            array = sitk.GetArrayFromImage(img)
            array = np.uint8(array > 80)*255
            cv2.imwrite(trf_dir + 'trf{}.png'.format(t + 1), array)
        elif t == label_t:
            array = np.uint8(label > 80)*255
            cv2.imwrite(trf_dir + 'trf{}.png'.format(t + 1), array)