#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Powered by haobo, 2021-10-08 11:03:16

import os
import cv2
import numpy as np

drawing=False   # true if mouse is pressed
mode=True

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# mouse callback function
def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, image, image2, mask

    # image2 = image.copy()
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                image = image2
                image = np.uint8(image*0.8) + cv2.line(mask,(current_former_x,current_former_y),(former_x,former_y),(0,0,51),10)    # the last number represents thickness
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(mask,(current_former_x,current_former_y),(former_x,former_y),(0, 0,51), 10)     # the last number represents thickness
            image = image2
            image = np.uint8(image*0.8) + mask
            current_former_x = former_x
            current_former_y = former_y
    elif event==cv2.EVENT_MBUTTONDOWN:
        mask *= 0
        image = image2
    return former_x,former_y

case= 95
start_time = 34
start_slice = 16
end_slice= 1

stack_dir     =  f"/Users/nureizzatyhamzaid/Desktop/FYP_23/Disease/Case {case}/Pre_Slices/paras/" #f"/home/haobo/Documents/UROP2021/16. 08Oct2019P3/time{t:03d}/"
print(stack_dir) 
mkdir(stack_dir + 'Segmented/')
mkdir(stack_dir + 'masked/')
image_dir     = stack_dir + "para{0}/time{1:03d}.png"
segmented_dir = stack_dir + "Segmented/Segmented Slice{0}time{1:03d}.png"
masked_dir    = stack_dir + 'masked/masked_slice{0}time{1:03d}.png'

 
    # gifs_dir= f"/Users/nureizzatyhamzaid/Library/Mobile Documents/com~apple~CloudDocs/FYP_23/Disease/Case {case}/Pre_Slices/gifs/para{s}.gif"
    # gifs_dir= r'/Users/nureizzatyhamzaid/Downloads/Your OxfordTube Ticket Confirmation - please print.pdf'
    # print()r"{}".format(gifs_dir)
    # os.system('"'+gifs_dir+'"')
    # print(time)
for s in range(start_slice, end_slice+1):
    print(image_dir.format(s,start_time))
    image = cv2.imread(image_dir.format(s,start_time))
    image2 = image.copy()
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.namedWindow(f"slice{s}time{start_time}")
    cv2.setMouseCallback(f"slice{s}time{start_time}", paint_draw)
    while 1:
        cv2.namedWindow(f'slice{s}time{start_time}', cv2.WINDOW_NORMAL)
        cv2.imshow(f"slice{s}time{start_time}",image)
        k=cv2.waitKey(1)& 0xFF
        # print(k)  
        if k==27: #Escape KEY
            cv2.imwrite(segmented_dir.format(s,start_time), mask[..., 2] * 5)
            cv2.imwrite(masked_dir.format(s,start_time), image)
            break
        elif k==114:
            image = image2
            mask *= 0
    cv2.destroyAllWindows()
