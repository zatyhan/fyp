from cv2 import cv2
import numpy as np
import os

start_slice = 13
end_slice   = 26
start_t     = 0
end_t       = 39
label_t     = 7
case_path   = '../6. 03Sep2019P2/'
paras_path  = '../6. 03Sep2019P2/paras/'
label_file  = 'label7.png'

for slice in range(start_slice, end_slice + 1):
    label = cv2.imread(paras_path + 'para{}_new/'.format(slice) + label_file,  0)
    # print(label)
    
    for t in range(start_t + 1, end_t + 1):
        pic_file  = case_path + 'time{0:03d}/'.format(t) + 'slice{0:03d}time{1:03d}.png'.format(slice, t)
        print(pic_file)
        pic = cv2.imread(pic_file)
        dst = pic[..., 2] + 50 * np.uint8(label == label.max())
        masked_path = paras_path + 'para{}_new/'.format(slice) + 'masked/'
        if not os.path.exists(masked_path):
            os.makedirs(masked_path)
        print(masked_path)
        cv2.imwrite(masked_path + 'slice{0:03d}time{1:03d}.png'.format(slice, t), dst)