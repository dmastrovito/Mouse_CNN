#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:31:47 2022

@author: danamastrovito
"""
#exec(open("plot_features.py").read())
import sys
import os
basedir = "/allen/programs/mindscope/workgroups/tiny-blue-dot/mousenet/Mouse_CNN/"
sys.path.append(os.path.join(basedir,'cmouse'))
import torch
import torchvision
import torchvision.transforms as transforms
from cifar_config import *
from train_config import *
import network
import matplotlib.pyplot as plt
import numpy as np
from mousenet_complete_pool import MouseNetCompletePool
from numpy.random import default_rng

SEED = 8
rng = default_rng(SEED)

expdir = os.path.join(basedir,'cmouse/exps/cifar/myresults/')
#"mask_3_cifar10_LR_0.001_M_0.5_mousenet/42_83.62.pt",\
#         "recurrent/baseline/mask_3_cifar10_LR_0.005_M_0.5_mousenet/42_22.2.pt",
#         "recurrent/sampled/mask_3_cifar10_LR_0.001_M_0.5_mousenet/42_10.0.pt",
files = ["mask_3_cifar10_LR_0.001_M_0.5_mousenet/42_83.62.pt",
         "recurrent/baseline/mask_no_grad_zeroing_3_cifar10_LR_0.005_M_0.5_mousenet/42_13.94.pt",
         "recurrent/baseline/mask_3_cifar10_LR_0.005_M_0.5_mousenet/42_17.21.pt"]
files = [os.path.join(expdir,file) for file in files]
    


transform = transforms.Compose([ transforms.Resize(INPUT_SIZE[1:]),
                            transforms.ToTensor(),])


testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
        download=True, transform=transform)


loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=2)
plt.clf()
tl = iter(loader)
images, label  = next(tl)
image = np.moveaxis( np.uint8(images[0].numpy() *255),0,2)
plt.imshow(image)    
plt.savefig(os.path.join(expdir,'image.png')) 
     

for file in files:
outputdir = os.path.dirname(file)
chkpt = torch.load(file,map_location=torch.device('cpu')) #keys epoch, best_acc1, state_dict
if 'recurrent' in file:
    net = network.load_network_from_pickle(os.path.join(basedir,'network_complete_updated_number(3,64,64)_edited_sigma_recurrent.pkl'))
    recurrent = True
    step_range = (30,40)
    n_steps = rng.integers(low=step_range[0], high=step_range[1], size=1)[0]     
else:
    net = network.load_network_from_pickle(os.path.join(basedir,'network_complete_updated_number(3,64,64).pkl'))
    recurrent = False
    n_steps = None
mousenet = MouseNetCompletePool(net, recurrent = recurrent)
mousenet.load_state_dict(chkpt['state_dict'])
mousenet.eval()
with torch.no_grad():
    out,calc_graph = mousenet(images,n_steps = n_steps, return_calc_graph = True)
    for region in list(calc_graph.keys()):
        print(region, calc_graph[region].shape)
        nchans = calc_graph[region].shape[1] 
        #nrows = [nchans % i for i in np.arange(1,nchans)]
        #nrows = np.max(np.where(np.array(nrows) ==0)) + 1
        nrows = 6
        ncols = int(np.ceil(nchans/nrows))
        plt.clf()
        for i in range(nchans):
            plt.subplot(nrows, ncols,i + 1)
            plt.imshow(np.uint8(255*calc_graph[region][0][i]))
            plt.xticks(ticks = None)
            plt.yticks(ticks = None)
        f = plt.gca()
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir,region.replace("/","_")+'.png'),dpi = 300)
    plt.clf()
    plt.subplot(2,1,1)
    if type(chkpt['training_loss'][0]) == torch.Tensor:
        chkpt['training_loss'] = np.array([l.detach().numpy() for l in chkpt['training_loss']])
    plt.plot(chkpt['training_loss'])
    plt.title("Training Loss")
    plt.subplot(2,1,2)
    plt.plot(chkpt['validation_loss'])
    plt.title("Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(outputdir,"loss.png"),dpi= 300, transparent = False)
                


