#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:37:05 2022

@author: danamastrovito
"""
#exec(open("cmouse/test_retinotopy.py").read())

import sys
sys.path.append('cmouse/')
sys.path.append('mouse_cnn/')
sys.path.append('../mouse_connectivity_models')

from architecture import Architecture
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from matplotlib import colors
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
from cifar_config import *
from train_config import *


layers = ['2/3','4','5']
DATA_DIR = "data"
recurrent = True
architecture = Architecture(data_folder=DATA_DIR,recurrent =recurrent)

targets = architecture.targets.keys()
areas = []
for target in targets:
    lrep = [l for l in layers if l in target]
    areas.append(target.replace(lrep[0],""))
 
    
areas = list(set(areas))
areas.remove('VISp')
areas.insert(0,'VISp')



transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(INPUT_SIZE[1:]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


transform_train = transforms.Compose([ transforms.Resize(INPUT_SIZE[1:]),
                            transforms.ToTensor(),])


trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
        download=True, transform=transform_train)


train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=2)
plt.clf()
tl = iter(train_loader)
images, label  = next(tl)
image = np.vstack((np.uint8(images[0].numpy() *256),np.uint8(np.zeros((1,64,64))+255)))
im = Image.fromarray(np.moveaxis(image,0,2),mode='RGBA')
imcp = im.copy()
color_names = ['lightgrey','green','blue','purple','orange','teal','red']
rgb = [colors.to_rgba(color)[:-1] for color in color_names]
font_size = 8
font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansCondensed.ttf",font_size) 

masks = []
centers = []
for a,area in enumerate(areas):
    plt.clf()
    im = imcp
    e,m = architecture.get_retinotopic_extent(area)
    masks.append(m)
    color = rgb[a]
    color = tuple(256 * elem for elem in color)
    alpha = np.uint8(np.expand_dims(m*.45*256,0))
    color_im = np.uint8(np.stack((np.zeros((64,64))+color[0],np.zeros((64,64))+color[1],np.zeros((64,64))+color[2])))
    im2 = Image.fromarray(np.moveaxis(np.vstack((color_im,alpha)),0,2),mode='RGBA')
    #imr = np.stack((m*(8*(a+1)),m*(8*(a+1)),m*(8*(a+1))))
    #imr = np.moveaxis(imr,0,2)
    #im2 = Image.fromarray(imr,mode='RGB')
    #mask = Image.fromarray(np.stack((m,m,m)),mode='RGBA')
    im = Image.alpha_composite(im,im2)
    nz = np.where(m ==1)
    center = (int(np.min(nz[1])),int(np.min(nz[0])))
    centers.append(center)
    I1 = ImageDraw.Draw(im)
    I1.text(center, area, fill=(0, 0, 0),font = font)
    #im = im.resize((6400,6400))        
    print(area)
    im.save("_".join((area,'retinotopy.png')))



im = imcp
for a,area in enumerate(areas):
    m = masks[a]
    color = rgb[a]
    color = tuple(256 * elem for elem in color)
    alpha = np.uint8(np.expand_dims(m*.45*256,0))
    color_im = np.uint8(np.stack((np.zeros((64,64))+color[0],np.zeros((64,64))+color[1],np.zeros((64,64))+color[2])))
    im2 = Image.fromarray(np.moveaxis(np.vstack((color_im,alpha)),0,2),mode='RGBA')
    #imr = np.stack((m*(8*(a+1)),m*(8*(a+1)),m*(8*(a+1))))
    #imr = np.moveaxis(imr,0,2)
    #im2 = Image.fromarray(imr,mode='RGB')
    #mask = Image.fromarray(np.stack((m,m,m)),mode='RGBA')
    im = Image.alpha_composite(im,im2)
    
'''
for a,area in enumerate(areas):
    nz = np.where(masks[a] ==1)
    center =centers[a]
    I1 = ImageDraw.Draw(im)
    I1.text(center, area, fill=(0, 0, 0),font = font)
'''       
im = im.resize((6400,6400))        
im.save('retinotopy.png')
