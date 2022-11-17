# -*- coding: utf-8 -*-
# Power by haobo, 2021-09-15 09:54:28

import os
import SimpleITK as sitk
from motionSegmentation import simpleSolver
import shutil as sh
# use segmentatation for anchor
# first without anchor

#getCompoundTimeList is range(0, 40)
# anchor= [[11, 32, 'img file 11', 'img file 32']]

start_slice = 13
end_slice   = 26
anchor_1    = 7
anchor_2    = 30
start_t     = 0
end_t       = 39
label_t     = 6
paras_path  = '../6. 03Sep2019P2/paras/'
label_file  = 'label7.png'
for slice in range(start_slice, end_slice + 1):
    anchor= [[anchor_1 - 1 , anchor_2 - 1,paras_path + 'para{}_new/label{}.png'.format(slice, anchor_1), paras_path + 'para{}_new/label{}.png'.format(slice, anchor_2)]]
    savePath = paras_path + 'para{}_new/'.format(slice)
    if not os.path.exists(savePath + 'scale.txt'):
        sh.copyfile(paras_path + '../scale.txt', savePath + 'scale.txt')        
    simpleSolver(savePath,startstep=1,endstep=6,fileScale=None,getCompoundTimeList=range(start_t, end_t + 1),compoundSchemeList=None,fftLagrangian=True,pngFileFormat=None,period=None,maskImg=True,anchor=anchor,bgrid=30.,imgfmt='uint8', finalShape=None,fourierTerms=4,twoD=True)
    # change everything to appropriate dimensions before mp.image.save