#!/usr/bin/env python
# coding: utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2'
import copy
import torch.multiprocessing as mp
import argparse
import time
import random
from torchvision import transforms
from torch import nn
# from torch.hub import load_state_dict_from_url
import numpy as np
from torch.utils.data import Dataset, DataLoader
from log_dice_loss import *
from torch.utils.data.distributed import DistributedSampler
# from SegDataset import SegDataset
from res34_unet import Unet
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)
import cv2

x_train_dir = "data_all/train_f/"
y_train_dir = "data_all/train_m/"

x_val_dir  = "data_all/val_f/"
y_val_dir = "data_all/val_m/"

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
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)
            mask = torch.Tensor(mask).permute(2, 0, 1)
            
        return image, mask, self.images_fps[i]
        
    def __len__(self):
        return len(self.ids) 


def setup_seed(seed):
     os.environ['PYTHONHASHSEED'] =str(seed)
     torch.manual_seed(seed)
     # torch.cuda.manual_seed(seed)
     # torch.cuda.manual_seed_all(seed)
     # torch.backends.cudnn.deterministic = True
     # torch.backends.cudnn.benchmark = False 
     np.random.seed(seed)
     random.seed(seed)
     imgaug.random.seed(seed)
     # torch.backends.cudnn.deterministic = True

# seed = 60
# setup_seed(seed)
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import imgaug
# imgaug.random.seed(seed)
import albumentations as albu
import matplotlib.pyplot as plt

def CCE_loss(y_pred, y_true):
        y_pred_log = torch.log(y_pred)
        nll_loss = nn.NLLLoss(reduction="mean")
        loss = nll_loss(y_pred_log, y_true.argmax(dim=1))
        return loss

def CCE_loss_2():
    epsilon = 1.e-7
    def _loss(y_pred, y_true):
        # y_true = 
        # y_pred = y_pred / torch.sum(y_pred, dim=1, keepdim=True)
        y_pred = torch.clamp(y_pred, epsilon, 1-epsilon)
        
        # y_t = tf.multiply(y_true, y_pred)
        ce = y_true * torch.log(y_pred)
        ce_out = - torch.mean(ce)
        return ce_out
    return _loss

class total_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = smp.utils.losses.DiceLoss()
        self.CCE_loss = CCE_loss_2()
        # self.softmax = nn.Softmax(dim=1)
    

    def forward(self, y_pred, y_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        CCE_loss = self.CCE_loss(y_pred, y_true)
        return dice_loss + CCE_loss

@torch.no_grad()
def dice_metric(y_pred, y_true):
    _dice1 = utils.metrics.Fscore(threshold=0.5, ignore_channels=[1,2], mode="img_mean")
    _dice2 = utils.metrics.Fscore(threshold=0.5, ignore_channels=[0,2], mode="img_mean")
    dice_1 = _dice1(y_pred, y_true)
    dice_2 = _dice2(y_pred, y_true)
    return (dice_1.item(), dice_2.item())

def dice_metric2(y_pred, y_true):
    _dice = smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[0,1], mode="img_mean")
    dice2 = _dice(y_pred, y_true)
    return dice2.item()


# criterion = log_dice_loss()
criterion = log_dice_loss()

def train_step(x, y_true, optim, model):
    optim.zero_grad()
    # model.train()
    y_pred = model(x)
    loss= criterion(y_pred, y_true)
    # print(type(y_true))
    loss.backward()
    optim.step()
    optim.zero_grad()

    metric = dice_metric(y_pred, y_true)
    # metric=smp.utils.metrics.Fscore()(y_pred, y_true)
    return loss.item(), *metric

@torch.no_grad()
def val_step(x, y_true, model):
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    metric = dice_metric(y_pred, y_true)
    return loss.item(), *metric
        
def get_training_augmentation():
    train_transform = [

        # albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(contrast_limit=0 ,p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                #albu.RandomContrast(p=1),
                albu.RandomBrightnessContrast(brightness_limit=0, p=1),
                # albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(320, 320)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')




# In[12]:


# x_train_dir = "all_set_f_03Sep/"
# y_train_dir = "all_set_m_03Sep/"
# train_valid_set = SegDataset(x_train_dir, y_train_dir, classes=["myocardium", "chamber"], augmentation=get_training_augmentation(), preprocessing=pre_transform)


# In[13]:


import copy

def seed_workers(worker_id):
    # print(torch.initial_seed())
    worker_seed = torch.initial_seed() % 2**32
    imgaug.random.seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # print(worker_seed)

# seed_workers(1)
# weight_decay = 0

# model = Unet(attention=None, Dropout=Dropout).to(device)
# model = Unet(attention=None, Dropout=Dropout).to(device)

# model.load_state_dict(torch.load("model_iter80_noatt_acc8694+9266_seed100_nearest_n128.pth"))
# criterion = total_loss()

# from train import train

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args(args=[])

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10023'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch

def cleanup():
    dist.destroy_process_group()

def train(gpu, args):
   
# plt.imshow(val_set[0][1])
# print(train_set[0][2])
# print(train_set.dataset.augmentation)
# print(model)
    
    # print(EPOCHS)
    weight_decay = 1e-5
    EPOCHS = 300
    lr = 0.0005
    BATCH_SIZE = 32


    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.world_size,
    	rank=rank
    )

    seed = 420
    g = torch.Generator()
    g.manual_seed(seed + gpu) 
    device = gpu
    setup_seed(seed + gpu)


    pre_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    train_set = SegDataset(x_train_dir, y_train_dir, classes=["myocardium", "chamber"], augmentation=get_training_augmentation(), preprocessing=pre_transform)
    val_set   = SegDataset(x_val_dir, y_val_dir, classes=["myocardium", "chamber"], augmentation=None, preprocessing=pre_transform)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)
    train_dataloader = DataLoader(train_set, BATCH_SIZE, num_workers=1, worker_init_fn=seed_workers, generator=g, pin_memory=True, sampler=train_sampler)
    # train_dataloader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4, worker_init_fn=seed_workers, generator=g, pin_memory=True)
    val_dataloader = DataLoader(val_set, BATCH_SIZE, num_workers=1, worker_init_fn=seed_workers, generator=g, pin_memory=True, sampler=val_sampler)
    # val_dataloader = DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=4, worker_init_fn=seed_workers, generator=g, pin_memory=True)
    torch.cuda.set_device(gpu)
    model = Unet(in_ch=3)
    # model.load_state_dict(torch.load('/home/haobo/HaoboSeg-pytorch/fcn2s_weights/fcn_iter99_seed420_nearest_n64_final_drop.pth'))
    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model,)
    # optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    writer = SummaryWriter(comment=f"noAtt_seed{seed}_determ_nearest_n{BATCH_SIZE}_l{weight_decay}", flush_secs=1)
    optim.param_groups[0]['lr']=1e-4
    min_loss = 0
    no_improve_step = 0

    idx = 0
    train_start, train_end, eval_start, eval_end = [0, 0, 0, 0]
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # writer = None
    lambda1 = lambda epoch : (1 - epoch/EPOCHS) **0.9 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda1)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    # 	train_set,
    # 	num_replicas=args.world_size,
    # 	rank=rank
    # )
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        model.train()
        # train_losses, train_dice_0s, train_dice_1s = [], [], []
        train_losses, train_dice_0s, train_dice_1s = [], [], []
        val_losses, val_dice_0s, val_dice_1s = [], [], []
        if gpu==0:
            train_start = time.time()

        #training begins
        for idx, (x_batch_train, y_batch_train, _) in enumerate(train_dataloader):
            # train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.cuda(), y_batch_train.cuda())
            # train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.to(gpu, non_blocking=True), y_batch_train.to(gpu, non_blocking=True), optim, optim2, coarse_model, model)
            train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.to(gpu, non_blocking=True), y_batch_train.to(gpu, non_blocking=True), optim, model)
            train_losses.append(train_loss)
            train_dice_0s.append(train_dice_0)
            train_dice_1s.append(train_dice_1)
        
        # gather mean criteria
        train_loss = np.mean(train_losses)
        train_dice_0 = np.mean(train_dice_0s)
        train_dice_1 = np.mean(train_dice_1s)
        if gpu==0:
            train_end = time.time()
        train_step_t = (train_end - train_start) / (idx + 1)

        # track epochs with no improvements in training loss

        
        if gpu==0:
            print(f"EPOCH: {epoch} Train_loss:  {train_loss:.4f}, Train_dice_myo: {train_dice_0:.4f}, Train_dice_chamber: {train_dice_1:.4f}")
        model.eval()
        if gpu==0:
            eval_start = time.time()
        for idx, (x_batch_val, y_batch_val, _) in enumerate(val_dataloader):
            # train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.cuda(), y_batch_train.cuda())
            val_loss, val_dice_0, val_dice_1 = val_step(x_batch_val.to(device, non_blocking=True), y_batch_val.to(device, non_blocking=True), model)
            val_losses.append(val_loss)
            val_dice_0s.append(val_dice_0)
            val_dice_1s.append(val_dice_1)
        if gpu==0:
            eval_end = time.time()
        eval_step_t = (eval_end - eval_start) / (idx + 1)
        val_loss = np.mean(val_losses)
        val_dice_0 = np.mean(val_dice_0s)
        val_dice_1 = np.mean(val_dice_1s)
        scheduler.step()
        if gpu == 0:
            print(f"EPOCH: {epoch} Val_loss:  {val_loss:.4f},Val_dice_myo: {val_dice_0:.4f}, Val_dice_chamber:{val_dice_1:.4f}")
        # writer.add_scalar('Loss/train', train_loss, epoch)

        # Writes scores in this epoch to tensorboard
            if writer is not None:
                writer.add_scalars('Loss', {'train':train_loss,
                                    'validation': val_loss}, epoch)
                # writer.add_scalar('Loss/validation', val_loss, epoch)
                writer.add_scalars('Myocardium Dice', {'train': train_dice_0,
                                            'validation': val_dice_0}, epoch)
                writer.add_scalars('Chamber Dice', {'train': train_dice_1,
                                            'validation': val_dice_1}, epoch)
        if not (epoch + 1) % 20 and gpu==0:
            torch.save(model.module.state_dict(), f"res34_single_weights_whole/res_{epoch}_seed{seed}_nearest_n{BATCH_SIZE}_final_drop.pth")

        end_time = time.time()
        if gpu==0:
            print(f"Total_time for epoch {epoch:d}: {end_time - start_time:.3f}s, Train speed: {train_step_t * 1000:.3f}ms/step, Val_speed: {eval_step_t * 1000:.3f}ms/step")
    
    cleanup()  # destroys the process group


if __name__=="__main__":
    main()
    