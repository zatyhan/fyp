#!/usr/bin/env python
# coding: utf-8

import os
import copy
import torch.multiprocessing as mp
import argparse
import time
from cv2 import cv2
import random
import torch, torchvision, torchinfo
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision import transforms
from torch import nn
from typing import Type, Any, Callable, Union, List, Optional
# from torch.hub import load_state_dict_from_url
import numpy as np
from torch.utils.data import Dataset, DataLoader
from log_dice_loss import *
from torch.utils.data.distributed import DistributedSampler
from SegDataset import SegDataset
from res34_unet import Unet as seg_resnet34
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)


x_train_dir = "train_vals/all_set_f_cropped/"
y_train_dir = "train_vals/all_set_m_cropped/"

# In[2]:


# OK it seems backprop of bilinear upsampling is not deterministic


# In[3]:


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
import imgaug
# imgaug.random.seed(seed)
import albumentations as albu
import matplotlib.pyplot as plt


# In[ ]:





# In[4]:




# In[5]:


# In[6]:


from Global_Context import ContextBlock as Global_Context


class coarse_Unet(nn.Module):
    def __init__(
        self,
        classes=3,
        in_ch = 3
    ):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1_d = nn.Sequential(
            nn.Conv2d(in_ch, 15, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 15),
            nn.LeakyReLU(),
            nn.Conv2d(15, 30, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 30),
            nn.LeakyReLU()
        )
        self.conv2_d = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(30, 30, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 30),
            nn.LeakyReLU(),
            nn.Conv2d(30, 60, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 60),
            nn.LeakyReLU()
        )
        self.conv3_d = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(60, 60, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 60),
            nn.LeakyReLU(),
            nn.Conv2d(60, 120, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 120),
            nn.LeakyReLU()
        )

        self.conv4_d = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(120, 120, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 120),
            nn.LeakyReLU(),
            nn.Conv2d(120, 240, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 240),
            nn.LeakyReLU()
        )
        self.bottle_neck = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(240, 240, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 240),
            nn.LeakyReLU(),
            nn.Conv2d(240, 480, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 480),
            nn.LeakyReLU(),
        )
        self.conv4_u1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(480, 240, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 240),
            nn.LeakyReLU(),
        )
        self.conv4_u2 = nn.Sequential(
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(240, 240, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 240),
            nn.LeakyReLU(),
            nn.Conv2d(240, 120, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 120),
            nn.LeakyReLU(),
            Global_Context(120, 15)
        )
        self.conv3_u = nn.Sequential(
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(120, 120, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 120),
            nn.LeakyReLU(),
            nn.Conv2d(120, 60, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 60),
            nn.LeakyReLU(),
            Global_Context(60, 15)
        )
        self.conv2_u = nn.Sequential(
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(60, 60, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 60),
            nn.LeakyReLU(),
            nn.Conv2d(60, 30, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 30),
            nn.LeakyReLU(),
            Global_Context(30, 15)
        )
        self.conv1_u = nn.Sequential(

            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(30, 60, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 60),
            nn.LeakyReLU(),
            nn.Conv2d(60, 30, 3, bias=False, padding="same"),
            nn.GroupNorm(5, 30),
            nn.LeakyReLU(),
            Global_Context(30, 15)
        )
        self.semi_outconv3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(60, 3, 3, padding="same"),
            nn.Sigmoid()
        )
        self.semi_outconv2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(30, 3, 3, padding="same"),
            nn.Sigmoid(),
        )
        self.semi_outconv1 = nn.Sequential(
            nn.Conv2d(30, 3, 3, padding="same"),
            nn.Sigmoid()
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(9, 3, 1, padding="same"),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        x1_d = self.conv1_d(img)
        x2_d = self.conv2_d(x1_d)
        x3_d = self.conv3_d(x2_d)
        x4_d = self.conv4_d(x3_d)
        x5   = self.bottle_neck(x4_d)
        x4_u = self.conv4_u2(self.conv4_u1(x5) + x4_d)
        x3_u = self.conv3_u(self.upsample(x4_u) + x3_d)
        x2_u = self.conv2_u(self.upsample(x3_u) + x2_d)
        x1_u = self.conv1_u(self.upsample(x2_u) + x1_d)
        deep_sup1 = self.semi_outconv1(x1_u)
        # print(deep_sup1.shape)
        deep_sup2 = self.semi_outconv2(x2_u)
        deep_sup3 = self.semi_outconv3(x3_u)
        deep_sup = torch.cat([deep_sup1, deep_sup2, deep_sup3], dim=1)
        out = self.outconv(deep_sup)
        return out, deep_sup1, deep_sup2, deep_sup3


# In[7]:


# model1 = coarse_Unet(in_ch=3)
# torchinfo.summary(model1, input_size=(32, 3, 256, 256))


# In[8]:


def CCE_loss(y_pred, y_true):
        y_pred_log = torch.log(y_pred)
        nll_loss = nn.NLLLoss(reduction="mean")
        loss = nll_loss(y_pred_log, y_true.argmax(dim=1))
        return loss

def CCE_loss_2():
    epsilon = 1.e-7
    def _loss(y_pred, y_true):
        # y_true = 

        y_pred = y_pred / torch.sum(y_pred, dim=1, keepdim=True)
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
    _dice1 = smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[1,2], mode="img_mean")
    _dice2 = smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[0,2], mode="img_mean")
    dice_1 = _dice1(y_pred, y_true)
    dice_2 = _dice2(y_pred, y_true)
    return (dice_1.item(), dice_2.item())

def dice_metric2(y_pred, y_true):
    _dice = smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[0,1], mode="img_mean")
    dice2 = _dice(y_pred, y_true)
    return dice2.item()


# In[9]:





# 

# In[10]:




# In[ ]:





# In[11]:

criterion = log_dice_loss()

def train_step(x, y_true, optim, optim2, coarse_model, model):
    optim.zero_grad()
    # model.train()
    y_pred_coarse =  coarse_model(x)
    loss1 = criterion(y_pred_coarse, y_true)
    # loss1 = (loss1_o + loss1_d1 + loss1_d2 +  loss1_d3)
    loss1.backward()
    optim.step()
    optim.zero_grad()
    y_pred = model(torch.cat([y_pred_coarse.detach(), x], dim=1))
    loss2 = criterion(y_pred, y_true)

    # loss2 = (loss2_o + loss2_d1 + loss2_d2 +  loss2_d3)
    # loss = loss1 + loss2
    loss2.backward()
    optim2.step()
    optim2.zero_grad()
    metric = dice_metric(y_pred, y_true)
    # metric=smp.utils.metrics.Fscore()(y_pred, y_true)
    return loss1.item() + loss2.item(), *metric

@torch.no_grad()
def val_step(x, y_true, coarse_model, model):
    y_pred_coarse =  coarse_model(x)
    loss1 = criterion(y_pred_coarse, y_true)
    # loss1 = (loss1_o + loss1_d1 + loss1_d2 +  loss1_d3)
    y_pred = model(torch.cat([y_pred_coarse.detach(), x], dim=1))
    loss2 = criterion(y_pred, y_true)
    
    # loss2 = (loss2_o + loss2_d1 + loss2_d2 +  loss2_d3)
    metric = dice_metric(y_pred, y_true)
    # metric = dice_metric2(y_pred, y_true)
    return loss1.item() + loss2.item(), *metric
        
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
                albu.HueSaturationValue(p=1),
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



# print(train_valid_set)


# In[62]:


# from train import train

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=4, type=int,
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


# In[63]:


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
    lr = 1e-4
    BATCH_SIZE = 64
    # Dropout = 0.2
    Dropout = 0


    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.world_size,
    	rank=rank
    )

    seed = 60
    g = torch.Generator()
    g.manual_seed(seed + gpu) 
    device = gpu
    setup_seed(seed + gpu)


    # pre_transform = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])])
    pre_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0],
                                 std=[1, 1, 1])])

    train_valid_set = SegDataset(x_train_dir, y_train_dir, classes=["myocardium", "chamber"], augmentation=get_training_augmentation(), preprocessing=pre_transform)
    train_len = len(train_valid_set)//5*4
    train_set, val_set = torch.utils.data.random_split(train_valid_set, [train_len, len(train_valid_set) - train_len])
    train_set.dataset = copy.deepcopy(train_valid_set)

    val_set.dataset.augmentation=None
    # train_set.dataset.augmentation=None
    train_set.dataset.augmentation=get_training_augmentation()
    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)
    train_dataloader = DataLoader(train_set, BATCH_SIZE, num_workers=1, worker_init_fn=seed_workers, generator=g, pin_memory=True, sampler=train_sampler)
    # train_dataloader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4, worker_init_fn=seed_workers, generator=g, pin_memory=True)
    val_dataloader = DataLoader(val_set, BATCH_SIZE, num_workers=1, worker_init_fn=seed_workers, generator=g, pin_memory=True, sampler=val_sampler)
    # val_dataloader = DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=4, worker_init_fn=seed_workers, generator=g, pin_memory=True)
    torch.cuda.set_device(gpu)
    coarse_model = seg_resnet34(in_ch=3, pretrained=False, Dropout=0)
    coarse_model.cuda(gpu)
    model = seg_resnet34(in_ch=6, pretrained=False, Dropout=0)
    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model,)
    coarse_model = nn.parallel.DistributedDataParallel(coarse_model, device_ids=[gpu], output_device=gpu)
    coarse_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(coarse_model,)
    optim = torch.optim.Adam(coarse_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    optim2 = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    writer = SummaryWriter(comment=f"noAtt_seed{seed}_determ_nearest_n{BATCH_SIZE}_l{weight_decay}", flush_secs=1)
    optim.param_groups[0]['lr']=1e-4

    idx = 0
    train_start, train_end, eval_start, eval_end = [0, 0, 0, 0]
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # writer = None
    lambda1 = lambda epoch : (1 - epoch/EPOCHS) **0.9 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda1)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optim2, lr_lambda=lambda1)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    # 	train_set,
    # 	num_replicas=args.world_size,
    # 	rank=rank
    # )
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        model.train()
        coarse_model.train()
        train_losses, train_dice_0s, train_dice_1s = [], [], []
        val_losses, val_dice_0s, val_dice_1s = [], [], []
        # train_losses, train_dice_2s = [], []
        # val_losses, val_dice_2s = [], []
        if gpu==0:
            train_start = time.time()

        #training begins
        for idx, (x_batch_train, y_batch_train, _) in enumerate(train_dataloader):
            # train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.cuda(), y_batch_train.cuda())
            train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.to(gpu, non_blocking=True), y_batch_train.to(gpu, non_blocking=True), optim, optim2, coarse_model, model)
            # train_loss, train_dice_2 = train_step(x_batch_train.to(gpu, non_blocking=True), y_batch_train.to(gpu, non_blocking=True), optim, optim2, coarse_model, model)
            train_losses.append(train_loss)
            train_dice_0s.append(train_dice_0)
            train_dice_1s.append(train_dice_1)
            # train_dice_2s.append(train_dice_2)
        
        # gather mean criteria
        train_loss = np.mean(train_losses)
        # train_dice_0 = np.mean(train_dice_0s)
        # train_dice_1 = np.mean(train_dice_1s)
        if gpu==0:
            train_end = time.time()
        train_step_t = (train_end - train_start) / (idx + 1)

        # track epochs with no improvements in training loss

        
        if gpu==0:
            print(f"EPOCH: {epoch} Train_loss:  {train_loss:.4f}, Train_dice_myo: {train_dice_0:.4f}, Train_dice_chanmber: {train_dice_1:.4f}, lr: {optim.param_groups[0]['lr']}")
    
        model.eval()
        coarse_model.eval()
        if gpu==0:
            eval_start = time.time()
        for idx, (x_batch_val, y_batch_val, _) in enumerate(val_dataloader):
            # train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.cuda(), y_batch_train.cuda())
            val_loss, val_dice0, val_dice1 = val_step(x_batch_val.to(device, non_blocking=True), y_batch_val.to(device, non_blocking=True), coarse_model, model)
            val_losses.append(val_loss)
            val_dice_0s.append(val_dice0)
            val_dice_1s.append(val_dice1)
            # val_dice_2s.append(val_dice2)
        if gpu==0:
            eval_end = time.time()
        eval_step_t = (eval_end - eval_start) / (idx + 1)
        val_loss = np.mean(val_losses)
        val_dice_0 = np.mean(val_dice_0s)
        val_dice_1 = np.mean(val_dice_1s)
        # val_dice_2 = np.mean(val_dice_2s)
        scheduler.step()
        scheduler2.step()
        if gpu == 0:
            print(f"EPOCH: {epoch} Val_loss:  {val_loss:.4f},Val_dice_myo: {val_dice_0:.4f}, Val_dice_chamber: {val_dice_1:.4f}")
        # writer.add_scalar('Loss/train', train_loss, epoch)

        # Writes scores in this epoch to tensorboard
            if writer is not None:
                writer.add_scalars('Loss', {'train':train_loss,
                                            'validation': val_loss}, epoch)
                # writer.add_scalar('Loss/validation', val_loss, epoch)
                writer.add_scalars('Myo Dice', {'train': train_dice_0,
                                                    'validation': val_dice_0}, epoch)
                writer.add_scalars('Chamber Dice', {'train': train_dice_1,
                                                    'validation': val_dice_1}, epoch)
            
        if not (epoch + 1) % 20 and gpu==0:
            torch.save(coarse_model.module.state_dict(), f"coarse_model1_iter{epoch}_seed{seed}_nearest_n{BATCH_SIZE}_final_drop.pth")
            torch.save(model.module.state_dict(), f"fine_model1_iter{epoch}_seed{seed}_nearest_n{BATCH_SIZE}_final_drop.pth")

        end_time = time.time()
        if gpu==0:
            print(f"Total_time for epoch {epoch:d}: {end_time - start_time:.3f}s, Train speed: {train_step_t * 1000:.3f}ms/step, Val_speed: {eval_step_t * 1000:.3f}ms/step")
    
    cleanup()  # destroys the process group

# In[64]:

if __name__=="__main__":
    main()
    

# In[ ]:
"""

x_test_dir = "test_f/"
y_test_dir = "test_m/"
test_set = SegDataset(x_test_dir, y_test_dir, classes=["myocardium",
 "chamber"], augmentation=None, preprocessing=pre_transform)
test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=4)
len(test_set)


# In[36]:


# test_losses, test_dice_0s, test_dice_1s, img_list = [], [], [], [] 
test_losses, test_dice_2s, img_list = [], [], []
for idx, (x_batch_test, y_batch_test, _) in enumerate(test_loader):
    model.eval()
        # train_loss, train_dice_0, train_dice_1 = train_step(x_batch_train.cuda(), y_batch_train.cuda())
    test_loss, test_dice2 = val_step(x_batch_test.to(device, non_blocking=True), y_batch_test.to(device, non_blocking=True))
    test_losses.append(test_loss)
    # test_dice_0s.append(test_dice0)
    # test_dice_1s.append(test_dice1)
    test_dice_2s.append(test_dice2)

print(f"Test_loss:  {np.mean(test_losses):.4f}, Test_dice_whole: {np.mean(test_dice_2s):.4f}")
# print(prof.table())


# In[53]:


import imageio

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

mkdir("sample_tests/test_f_cropped")
model.eval()
img_list = []



# In[44]:
"""