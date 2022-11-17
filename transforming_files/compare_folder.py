import os

masks = os.listdir('/home/haobo/Documents/UROP2021/Truth_pics_19Jul/all_set_m/')
frames = os.listdir('/home/haobo/Documents/UROP2021/Truth_pics_19Jul/all_set_f/')
diff = []

for m in frames:
    if m not in masks:
        diff.append(m)

print(diff)