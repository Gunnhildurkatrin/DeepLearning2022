#from skimage.measure import profile_line
#from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from os import listdir
from os.path import isfile, join
import nltk
import torch
import torch.nn as nn
#import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import os
from torchvision import datasets, transforms
from torch.nn import init

import torch.nn.functional as F

from tqdm import tqdm
import torch.optim as optim
import cv2
from skimage import color, io, measure, img_as_ubyte, draw
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import matrix_transform
from skimage.filters import gaussian
from skimage import exposure
import math
from skimage.transform import rescale,resize,downscale_local_mean
import random

import seaborn as sns
import sklearn
from sklearn import metrics

#Hyperparameters

LEARNING_RATE = 0.0001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 75
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

PATH = 'FinalRuns2/UNET3plus_normal/'
model_name = 'UNET_3plus'

torch.cuda.empty_cache()


train_ids_photos = ['71_a.jpg', '412.jpg', '283_a.jpg', '68_a.jpg', '95_a.jpg',
       '201.jpg', '363_a.jpg', '361.jpg', '62.jpg', '74.jpg', '57_a.jpg',
       '281_a.jpg', '409_a.jpg', '199_a.jpg', '121_a.jpg', '73_a.jpg',
       '59.jpg', '341_a.jpg', '158.jpg', '125_a.jpg', '224_a.jpg',
       '381_a.jpg', '285_a.jpg', '64.jpg', '8.jpg', '66.jpg', '557_a.jpg',
       '211.jpg', '75_a.jpg', '801_a.jpg', '73.jpg', '9.jpg', '471.jpg',
       '104_a.jpg', '141_a.jpg', '120_a.jpg', '72_a.jpg', '379_a.jpg',
       '795_a.jpg', '158_a.jpg', '114.jpg', '29.jpg', '15.jpg', '116.jpg',
       '499.jpg', '54_a.jpg', '773_a.jpg', '301.jpg', '103.jpg',
       '94_a.jpg', '16.jpg', '12.jpg', '102_a.jpg', '259.jpg',
       '417_a.jpg', '13.jpg', '74_a.jpg', '39.jpg', '124_a.jpg',
       '535_a.jpg', '272.jpg', '428_a.jpg', '399_a.jpg', '475.jpg',
       '313.jpg', '38.jpg', '108.jpg', '120.jpg', '699_a.jpg', '493.jpg',
       '664_a.jpg', '518.jpg', '334_a.jpg', '121.jpg', '34.jpg',
       '87_a.jpg', '78_a.jpg', '241.jpg', '312_a.jpg', '308.jpg',
       '37.jpg', '33.jpg', '65_a.jpg', '137_a.jpg', '536.jpg',
       '353_a.jpg', '331.jpg', '26.jpg', '32.jpg', '156_a.jpg',
       '113_a.jpg', '455.jpg', '290.jpg', '83_a.jpg', '351_a.jpg',
       '130.jpg', '519_a.jpg', '31.jpg', '42.jpg', '180.jpg', '582_a.jpg',
       '153_a.jpg', '209.jpg', '79_a.jpg', '606_a.jpg', '60_a.jpg',
       '5.jpg', '57.jpg', '7.jpg', '41.jpg', '69.jpg', '82.jpg',
       '700_a.jpg', '62_a.jpg', '249_a.jpg', '816_a.jpg', '86_a.jpg',
       '237.jpg', '724_a.jpg', '182.jpg', '114_a.jpg', '54.jpg', '6.jpg',
       '192.jpg', '44.jpg', '93.jpg', '798_a.jpg', '66_a.jpg', '45.jpg',
       '193.jpg', '855_a.jpg', '84.jpg', '64_a.jpg', '224.jpg',
       '99_a.jpg', '706_a.jpg', '112_a.jpg', '85.jpg', '147.jpg']


#create a list of all train ids and test ids 
p='clean_data'
train_ids_doors = []
for file in os.listdir(p):
    if file.startswith("DOOR"):
        train_ids_doors.append(file)



#Comment out train_ids_doors to include more data
train_ids = train_ids_photos # + train_ids_doors 


test_ids = ['0_a.jpg','1_a.jpg','2_a.jpg','3_a.jpg','5_a.jpg','6_a.jpg','10_a.jpg','11_a.jpg','12_a.jpg','19_a.jpg','20_a.jpg','21_a.jpg','22_a.jpg','24_a.jpg','26_a.jpg','28_a.jpg','29_a.jpg','32_a.jpg','33_a.jpg','35_a.jpg','36_a.jpg','39_a.jpg','40_a.jpg','43_a.jpg','45_a.jpg','46_a.jpg','47_a.jpg','50_a.jpg','51_a.jpg','52_a.jpg']
#remove test_ids from the train_ids

    
test_ids = [s.replace('jpg', 'npy') for s in test_ids] # use clean data
train_ids = [s.replace('jpg', 'npy') for s in train_ids] # use clean data 
train_ids = [s.replace('png', 'npy') for s in train_ids] # use clean data

for i in test_ids:
    if i in train_ids:
        train_ids.remove(i)

random.shuffle(train_ids)

val_ids = train_ids[:len(train_ids)//4] #25 percent is train  data
train_ids = train_ids[len(train_ids)//4:] # 75 percent is test data 

path = '/zhome/88/1/165903/DeepLearning/clean_data/'

class ImageDataset(Dataset): # call AugmentingDataset
    def __init__(self, path, img_ids, rotate60, rotate20, gaussian,gamma, log_con,zooming ):
        self.path = path
        self.img_ids = img_ids
        #self.transform = transform
        self.rotate60 = rotate60
        self.rotate20 = rotate20
        self.gaussian = gaussian
        self.gamma = gamma
        self.log_con = log_con
        self.zooming = zooming

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        data = np.load(os.path.join(self.path, img_id))

        img = data[0:3].transpose(1,2,0)
        im_gray = color.rgb2gray(img)
        im_byte = img_as_ubyte(im_gray)
        clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
        img = clahe.apply(im_byte)
        
        if self.rotate60:
            img = rotate(img, 60, mode="reflect")
        if self.rotate20:
            img = rotate(img, 20, mode="reflect")
        if self.gaussian:
            img = gaussian(img, 1)
        if self.gamma:
            img = exposure.adjust_gamma(img, 2)
        if self.log_con:
            img = exposure.adjust_log(img, 1)
        if self.zooming:
            img = img[50:200,50:200]
            scale_factor = 256 / (200-50)
            img = rescale(img, scale_factor, anti_aliasing=True)
            img = img_as_ubyte(img)

            
         
        mask = data[-1]
        
        if self.rotate60:
            mask = rotate(mask, 60, mode="reflect")
        if self.rotate20:
            mask = rotate(mask, 20, mode="reflect")
        if self.gaussian:
            mask = gaussian(mask, 1)
        if self.zooming:
            mask = mask[50:200,50:200]
            scale_factor = 256 / (200-50)
            mask = rescale(mask, scale_factor, anti_aliasing=True)
        #the mask doesn't need to be changed for gamma and log 
        
        n_classes = 9 #len(np.unique(mask))
        

        img = torch.Tensor(img)
        img = img.type(torch.int64)
        img = torch.unsqueeze(img, 0)
        
        #print(mask.shape)
        mask = torch.Tensor(mask)
        mask = mask.type(torch.int64)
        #print(mask.shape)
        mask = F.one_hot(mask, num_classes=n_classes)
        mask = torch.permute(mask, (2, 0,1))


        return {'image': torch.Tensor(img.float()), 'mask': torch.Tensor(mask.float())}

    def __len__(self):
        return len(self.img_ids)


train_dataset = ImageDataset(path, train_ids, False,False,False,False,False,False)
train_dataset_rotate60 = ImageDataset(path, train_ids, True, False, False,False, False,False)
train_dataset_rotate20 = ImageDataset(path, train_ids, False, True, False,False, False,False)
train_dataset_gaussian = ImageDataset(path, train_ids, False, False, True,False, False,False)
train_dataset_gamma = ImageDataset(path, train_ids, False, False, False,True,False, False)
train_dataset_log = ImageDataset(path, train_ids, False, False, False,False, True, False)
train_dataset_zooming = ImageDataset(path, train_ids, False, False, False,False, False,True)

val_dataset = ImageDataset(path, val_ids, False,False,False,False,False,False)
val_dataset_rotate60 = ImageDataset(path, val_ids, True, False, False,False, False,False)
val_dataset_rotate20 = ImageDataset(path, val_ids, False, True, False,False, False,False)
val_dataset_gaussian = ImageDataset(path, val_ids, False, False, True,False, False,False)
val_dataset_gamma = ImageDataset(path, val_ids, False, False, False,True,False,False)
val_dataset_log = ImageDataset(path, val_ids, False, False, False,False, True,False)
val_dataset_zooming = ImageDataset(path, val_ids, False, False, False,False, False,True)

# comment out train_dataset_zooming to include zoom augmentation
train_dataset = train_dataset + train_dataset_rotate60 + train_dataset_rotate20 + train_dataset_gaussian + train_dataset_gamma + train_dataset_log# +train_dataset_zooming
val_dataset = val_dataset + val_dataset_rotate60 + val_dataset_rotate20 + val_dataset_gaussian + val_dataset_gamma + val_dataset_log # +val_dataset_zooming
test_dataset = ImageDataset(path, test_ids, False,False,False,False,False,False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)



for batch in train_loader:
    b = batch
    break

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class UNet_3Plus(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True)  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True)  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear',align_corners=True)  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        return d1

def accuracy(target, pred):
    
    
    num_correct = (pred==target).sum()
    num_pixels = torch.numel(pred)

    return (num_correct/num_pixels)*100


def dice_coeff(input, target, reduce_batch_first = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input, target, reduce_batch_first= False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input, target, multiclass= False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


model = UNet_3Plus(in_channels=1,n_classes=9).to(DEVICE)#.to(DEVICE)
# If we want to run using pre trained weights: https://pytorch.org/tutorials/beginner/saving_loading_models.html

# model.load_state_dict(torch.load('WeightsUNET_3plus1.pt'))
# model.eval()

#loss_fn = nn.CrossEntropyLoss(weight=class_sample_count,reduction='mean') #BCEWithLogitsLoss() for binary classification
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

lambda1 = lambda epoch: 0.65 ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
lrs=[]


validation_every_steps = math.ceil(len(train_dataset)/BATCH_SIZE)

step = 0
model.train()

train_accuracies = []
valid_accuracies = []


for epoch in range(NUM_EPOCHS):
    print('epoch no:',epoch)
    
    train_accuracies_batches = []
    
    for d in train_loader:
        
        
        inputs=d['image'].to(DEVICE).float()
        targets = d['mask'].to(DEVICE).float()
                        
        # Forward pass, compute gradients, perform one training step.
        # Your code here!
        output = model(inputs)
        output=F.softmax(output)
    
        
        # Compute loss.
        loss = dice_loss(output,targets,multiclass=True) +loss_fn(output, targets)
    
        
        # Clean up gradients from the model.
        optimizer.zero_grad()
        
        # Compute gradients based on the loss from the current batch (backpropagation).
        loss.backward()
        
        # Take one optimizer step using the gradients computed in the previous step.
        optimizer.step()
        
        lrs.append(optimizer.param_groups[0]["lr"])
        #scheduler.step()

        # Increment step counter
        step += 1
        
        # Compute accuracy.
        
        predictions=torch.argmax(output,dim=1)
        targets = torch.argmax(targets,dim=1)

        #print(torch.unique(predictions))
        train_accuracies_batches.append(accuracy(targets, predictions))
        
        #print('train acc:',train_accuracies_batches[-1], 'val_acc: ', valid_accuracies_batches[-1])
        
        if step % validation_every_steps == 0:
            
            # Append average training accuracy to list.
            train_accuracies.append(torch.mean(torch.Tensor(train_accuracies_batches)))
            
            train_accuracies_batches = []
        
            # Compute accuracies on validation set.
            valid_accuracies_batches = []
            with torch.no_grad():
                model.eval()
                for d in val_loader:
                    
                    inputs=d['image'].to(DEVICE).float()
                    targets = d['mask'].to(DEVICE).float()
                    
                    output = model(inputs)
                    output = F.softmax(output)
        
                    loss = dice_loss(output,targets,multiclass=True) +loss_fn(output, targets)
                    
                    predictions=torch.argmax(output,dim=1)
                    targets = torch.argmax(targets,dim=1)

                    print(torch.unique(predictions))

                    # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                    valid_accuracies_batches.append(accuracy(targets, predictions))
                    
                 
                model.train()
                
            # Append average validation accuracy to list.
            valid_accuracies.append(torch.mean(torch.Tensor(valid_accuracies_batches)))
            
            x = np.arange(0, step/validation_every_steps, 1).tolist()
            plt.plot(x,train_accuracies)
            plt.plot(x,valid_accuracies)
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy (%)')
            plt.show()


    if (epoch%5)==0:
        torch.save(model.state_dict(), './WeightsUNET_3plus.pt')

     
            #print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
            #print(f"             test accuracy: {valid_accuracies[-1]}")

print("Finished training. , total steps:",step)

#x = np.arange(1, step/validation_every_steps, 1).tolist()
#plt.plot(x,train_accuracies)
#plt.plot(x,valid_accuracies)
#plt.show()

#print(train_accuracies)
#print(valid_accuracies)

data = iter(test_loader).next()
inputs = data['image'].to(DEVICE).float()
print(input)
targets = data['mask'].to(DEVICE).float()

outputs = model(inputs)
pred = F.softmax(outputs)
predictions = torch.argmax(pred, dim=1)

def plotting(no):
    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    plt.imshow(inputs[no].permute(1,2,0).cpu())
    
    plt.subplot(1,3,2)
    plt.imshow(predictions[no].cpu())
    
    plt.subplot(1,3,3)
    maski = targets[no].permute(1,2,0).cpu()
    maski = torch.argmax(maski, dim=2)
    plt.imshow(maski)
    
    plt.savefig(PATH+'Fig_ind{}_{}'.format(no,model_name))


x = np.arange(0, step/validation_every_steps, 1)#.tolist()
plt.plot(x,train_accuracies,label='Train accuracy')
plt.plot(x,valid_accuracies,label='Validation accuracy')
plt.legend()
plt.savefig(PATH+'Accuracy_'+model_name)

with open(PATH+'ACC_LIS_'+model_name+'.txt', 'w') as fp:
    for item in valid_accuracies:
        # write each item on a new line
        fp.write("%s\n" % item)



plotting(0)
plotting(1)
plotting(2)
plotting(3)
plotting(4)
plotting(5)
plotting(6)
plotting(7)
plotting(8)
plotting(9)
plotting(10)
plotting(11)
plotting(12)
plotting(13)
plotting(14)
plotting(15)

predictions[14].unique(return_counts=True)

maski = targets[14].permute(1,2,0)
maski = torch.argmax(maski, dim=2)

maski.unique(return_counts=True)



plt.figure()
plt.plot(range(len(lrs)),lrs,label='Learning Rate')
plt.ylabel('Learning Rate')
plt.xlabel('Iteration')
plt.title('Learning Rate')
plt.legend()
#plt.savefig('LearningRate_UNET')


def compute_confusion_matrix(target, pred, normalize=None):
    return metrics.confusion_matrix(
        target, 
        pred,
        normalize=normalize
    )

# Evaluate test set
confusion_matrix = np.zeros((9, 9))




with torch.no_grad():
    test_accuracies = []
    loss_vals=torch.zeros(2)
    model.eval()
    i=0
    for d in test_loader:
        inputs = d['image'].to(DEVICE).float()
        targets = d['mask'].to(DEVICE).float()
            

        output = model(inputs)
        loss = dice_loss(output,targets,multiclass=True)+ loss_fn(output, targets)
        loss_vals[i]=loss.float()
        i+=1
        predictions = output.max(1)[1]

        # Multiply by len(inputs) because the final batch of DataLoader may be smaller (drop_last=True).
        #np.array(predictions.reshape(-1,256))
        #np.array(targets.reshape(-1,256))
        targets = torch.argmax(targets, dim=1)
        test_accuracies.append(accuracy(targets, predictions))
        LEN = predictions.shape[0]
        predictions = np.array(predictions.cpu().reshape(LEN*256*256))
        targets = np.array(targets.cpu().reshape(LEN*256*256))

        
        confusion_matrix += compute_confusion_matrix(targets, predictions)

    test_accuracy = np.sum(test_accuracies) / len(test_dataset)

  
    
    model.train()

with open(PATH+'TEST_ACC_LIS_'+model_name+'.txt', 'w') as fp:
    for item in test_accuracies:
        # write each item on a new line
        fp.write("%s\n" % item)

def normalize(matrix, axis):
    axis = {'true': 1, 'pred': 0}[axis]
    return matrix / matrix.sum(axis=axis, keepdims=True)

mean_acc = np.mean(np.diag(normalize(confusion_matrix, 'true')))
mean_loss=torch.mean(loss_vals)

with open(PATH+'UNETavg_test_acc.txt', 'w') as f:
  f.write('Average class accuracy: {}\n Average Loss: {}'.format( mean_acc,mean_loss))

x_labels = ["Background", "Front door", "Back door", "Front fender", \
    "Car side body", "Front bumper","Hood", "Rear bumper ", "Trunk"]

with sns.axes_style('whitegrid'):
    plt.figure(figsize=(8, 8))
    sns.barplot(x=x_labels, y=np.diag(normalize(confusion_matrix, 'true')))
    plt.xticks(rotation=35, fontsize=7)
    plt.title("Per-class accuracy")
    plt.ylabel("Accuracy")
    plt.savefig(PATH+"ClassAccs"+model_name+'.png')
    