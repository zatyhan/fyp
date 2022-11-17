import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class SegDataset(Dataset):
    def __init__(
        self, 
        images_dir, 
        masks_dir, 
        classes=None, 
        augmentation=None, 
        preprocessing=None,
    ):
        self.CLASSES = ["myocardium", "chamber"]
        images_dirs = os.listdir(images_dir)
        # np.random.shuffle(images_dirs)
        self.ids = images_dirs
        # self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) + 1 for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values ]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        if mask.shape[-1] != 1:
            # background = 1 - mask.sum(axis=-1, keepdims=True)
            whole = mask.sum(axis=-1, keepdims=True)
            # mask = np.concatenate((mask, background), axis=-1)
            mask = np.concatenate((mask, whole), axis=-1)
        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)
            mask = torch.Tensor(mask).permute(2, 0, 1)
            
        return image, mask, self.images_fps[i]
        
    def __len__(self):
        return len(self.ids)
