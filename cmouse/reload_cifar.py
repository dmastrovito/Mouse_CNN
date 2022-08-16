#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 10:28:59 2022

@author: danamastrovito
"""

from cifar_config import *
from train_config import *
from train_cifar import test, train, debug_memory, adjust_learning_rate, get_data_loaders,set_save_dir
import torch
import torch.optim as optim
import network
import os
import sys
from mousenet_complete_pool import MouseNetCompletePool

basedir = "/allen/programs/mindscope/workgroups/tiny-blue-dot/mousenet/Mouse_CNN/"
sys.path.append(os.path.join(basedir,'cmouse'))


#file = os.path.join(basedir,'cmouse/exps/cifar/myresults',"mask_3_cifar10_LR_0.001_M_0.5_mousenet/42_80.34.pt")
file = os.path.join(basedir,'cmouse/exps/cifar/myresults/recurrent/sampled',"mask_3_cifar10_LR_0.001_M_0.5_mousenet/42_10.0.pt")


chkpt = torch.load(file) #keys epoch, best_acc1, state_dict
if 'recurrent' in file:
    net = network.load_network_from_pickle(os.path.join(basedir,'network_complete_updated_number(3,64,64)_edited_sigma_recurrent.pkl'))
    recurrent = True
    nsteps = "sampled"
    step_range = (30,40)
else:
    net = network.load_network_from_pickle(os.path.join(basedir,'network_complete_updated_number(3,64,64).pkl'))
    recurrent = False
    nsteps = None
    step_range = None


set_save_dir(recurrent = recurrent,nsteps = nsteps)
outputdir = os.path.dirname(file)
mousenet = MouseNetCompletePool(net, recurrent = recurrent)
mousenet.load_state_dict(chkpt['state_dict'])
epoch = chkpt['epoch']
config = None
device = torch.device("cuda")
mousenet.to(device)

train_loader, test_loader = get_data_loaders()
optimizer = optim.SGD(mousenet.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=5e-4)
best_acc = chkpt['best_acc']
validation_loss = chkpt['validation_loss']
training_loss = chkpt['training_loss']  

best_acc = test(config, mousenet, device, test_loader, epoch,best_acc = best_acc, training_loss=training_loss, validation_loss = validation_loss,\
                recurrent=recurrent,nsteps = nsteps,step_range = step_range)  

for epoch in range(epoch+1, EPOCHS + 1):  # loop over the dataset multiple times
    adjust_learning_rate(config, optimizer, epoch)
    print(epoch)  
    training_loss = train(config, mousenet, device, train_loader, optimizer, epoch,training_loss = training_loss,recurrent=recurrent,\
                          nsteps = nsteps,step_range = step_range)
    debug_memory()
    best_acc = test(config, mousenet, device, test_loader, epoch,best_acc = best_acc, training_loss=training_loss, validation_loss = validation_loss,\
                    recurrent=recurrentt,nsteps = nsteps,step_range = step_range)  
    

print('Finished Training')

if recurrent:
    if nsteps == 'sampled':
        outfile = "_".join()
    torch.save(mousenet.state_dict(),"mousenet_cifar_trained_recurrent.sav")
else:
    torch.save(mousenet.state_dict(),"mousenet_cifar_trained.sav")

    