# Creates gif for segmented labeled images

import imageio as im

def create_gif(masked_dir, slice, total_t, gif_name):
    frames = []
    for t in range(1, total_t + 1):
        im_dir = masked_dir + f'slice{slice:03d}time{t:03d}.png'
        img = im.imread(im_dir)
        frames.append(img)
    im.mimsave(gif_name, frames, 'GIF', duration=0.1)

if __name__ == '__main__':
    slice = 21
    total_t = 20
    masked_dir = '/home/haobo/Documents/UROP2021/16. 08Oct2019P3/paras/img_with_mask/'
    gif_name = masked_dir + f'../../slice{slice}.gif'
    print(gif_name)
    create_gif(masked_dir, slice, total_t, gif_name)