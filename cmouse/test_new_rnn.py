#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 19:04:11 2022

@author: danamastrovito
"""

#exec(open("cmouse/test_new_rnn.py").read())

import sys
sys.path.append('cmouse')

from cifar_config import *
from train_config import *
from train_cifar import  debug_memory, get_data_loaders,set_save_dir
import torch
import torch.optim as optim
import network
import mousenet_model
import os
import sys
import mousenet_model
from numpy.random import default_rng
import random
import numpy as np
from mousenet_model import *
import matplotlib.pyplot as plt

basedir = "/allen/programs/mindscope/workgroups/tiny-blue-dot/mousenet/Mouse_CNN/"

def plot_states(states):
    
    #plt.clf()
    for r,region in enumerate(list(states.keys())):
        plt.clf()
        #plt.subplot(4,6,r+1)
        plt.plot(np.arange(1,len(states[region])+1),np.array(states[region]),marker='.')
        plt.title(region)
        plt.savefig("_".join((region.replace("/","_"),'states_over_time.png')))
    #plt.tight_layout()
    #plt.savefig("states_over_time.png",dpi = 300)
    
    
def plot_loss(loss):
    plt.clf()
    plt.plot(loss)
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig("Training_loss.png")
        
#file = os.path.join(basedir,'cmouse/exps/cifar/myresults',"mask_3_cifar10_LR_0.001_M_0.5_mousenet/42_83.62.pt")
#file = os.path.join(basedir,'cmouse/exps/cifar/myresults/recurrent/sampled',"mask_3_cifar10_LR_0.001_M_0.5_mousenet/42_10.0.pt")
#file = os.path.join(basedir,'cmouse/exps/cifar/myresults/recurrent/sampled/max_tanh_multiplicative_recurrence_3_cifar10_LR_0.005_M_0.5_mousenet/42_57.72.pt')
#file = os.path.join(basedir,"cmouse/exps/cifar/myresults/recurrent/baseline/sigmoid_multiplicative_recurrence_3_cifar10_LR_0.05_M_0.5_mousenet/42_init.pt")
#file = os.path.join(basedir,'cmouse/exps/cifar/myresults/recurrent/sampled/ReLU_eachstep_multiplicative_recurrence_3_cifar10_LR_0.005_M_0.5_mousenet/42_37.95.pt')

recurrent = True

SEED = 42
rng = default_rng(SEED)
torch.backends.cudnn.deterministic = True
random.seed(SEED)       # python random seed
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED)    # numpy random seed

step_range= (30,40)

mask = 3
#net = network.load_network_from_pickle(os.path.join(basedir,'network_complete_updated_number(3,64,64)_edited_sigma_recurrent.pkl'))
net = network.load_network_from_pickle(os.path.join('recurrent_mousenet_inputsize32_convtranspose2d.pkl'))

device = torch.device('cuda')


mn= mousenet_model.mousenet(net, recurrent = recurrent,device=device)
train_loader, test_loader = get_data_loaders()


loss = nn.CrossEntropyLoss()



mn.reset(BATCH_SIZE, device)
mn.to(device)
params = list(mn.named_parameters())

#optimizer = optim.Adam(mn.parameters(),lr = 0.0001)
optimizer = optim.SGD(mn.parameters(),lr = 0.01)

mn.train()
'''
data, labels = next(iter(train_loader))
data, labels = data.to(device), labels.to(device)

mn.reset(BATCH_SIZE, device)
'''

'''
out  = mn(data,n_steps = 40)
l = loss(out, labels)
l.backward()

for i in range(len(mn.regions)):
    grads = [mn.regions[i].convs[key].conv[1].weight.grad for key in mn.regions[i].convs] 
    print(i,mn.regions[i].name, any(j != None for j in grads))
'''
Trainloss = []
Vloss = []
    
for epoch in range(1, EPOCHS + 1):
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        n_steps = rng.integers(low=step_range[0], high=step_range[1], size=1)[0]   
        out,states = mn(data,n_steps =n_steps,track_states = True)
        l = loss(out, labels)
        l.backward()
        
        optimizer.step()
        
        Trainloss.append(l.detach().cpu().numpy())
        mn.reset(BATCH_SIZE,device)
        if batch_idx %10 ==0:
            debug_memory()
            plot_states(states)
            plot_loss(Trainloss)
            for p in params:
                if p[1].grad != None:
                    print(p[0],p[1].grad.mean())
            for region in mn.regions:
                print(region.decay.grad.mean())
            print(n_steps,epoch,l.detach().cpu().numpy(),batch_idx, "Of",len(train_loader))
        
        '''
        if batch_idx %100 == 0:
            plt.clf()
            for i in range(net.layers):
                plot(states[i])
        '''
    with torch.no_grad():
      correct = 0
      for data, target in test_loader:
          # Load the input features and labels from the test dataset
          data, target = data.to(device), target.to(device)
          # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
          n_steps = rng.integers(low=step_range[0], high=step_range[1], size=1)[0]      
          output = mn(data, n_steps)                 
          # Compute the loss sum up batch loss
          #test_loss += F.nll_loss(output, target, reduction='sum').item()
          test_loss = loss(output, target)
          pred = output.max(1, keepdim=True)[1]
          correct += pred.eq(target.view_as(pred)).sum().item()         
          Vloss.append(test_loss.cpu().numpy())
      acc = 100. * correct / len(test_loader.dataset)
      print(acc)
        
      
'''       
debug_memory()


mn.connections[mn.regions[1].convs[0]].conv[0].weight.grad
'''


'''
Flayer= net.find_conv_source_target('VISp5', 'VISpor4')
Rlayer = net.find_conv_source_target('VISpor4', 'VISp5')

from mousenet_model import *
#padding = int((Rlayer.params.kernel_size - Rlayer.params.in_size + Rlayer.params.out_size - Rlayer.params.stride)/2) #if output_padding is 1
#padding = int((Rlayer.params.kernel_size - Rlayer.params.in_size + Rlayer.params.out_size -1 - Rlayer.params.stride)/2) #if output_padding is 0
padding = int((Rlayer.params.out_size - Rlayer.params.in_size)/2)

F = Conv2dMask(Flayer.params.in_channels, Flayer.params.out_channels, Flayer.params.kernel_size, Flayer.params.gsh,Flayer.params.gsw, \
               stride=Flayer.params.stride,mask=3, padding=Flayer.params.padding,padding_mode= Flayer.params.padding_mode)

data = torch.rand(1,32,64,64)
Ftest = F(data) #torch.Size([1, 5, 32, 32])
R = RConv2dMask(Rlayer.params.in_channels, Rlayer.params.out_channels, Rlayer.params.kernel_size, Rlayer.params.gsh,Rlayer.params.gsw, \
               stride=Rlayer.params.stride,mask=3, padding=padding,padding_mode= Rlayer.params.padding_mode)

Rtest = R(Ftest)
nnp = nn.ReflectionPad2d(padding)  
padded_in= nnp(Ftest) 
ConvT = nn.ConvTranspose2d(Rlayer.params.in_channels, Rlayer.params.out_channels, Rlayer.params.kernel_size,stride=2)
ConvT = nn.ConvTranspose2d(Rlayer.params.in_channels, Rlayer.params.out_channels, Rlayer.params.kernel_size,stride=2,dilation=1,padding=3)
ConvT(padded_in)
'''

'''
connections = nn.ModuleDict()
resolutions = {}
for layer in net.layers[1:]:
    layer_name = layer_name = "_".join((layer.source_name,layer.target_name))
    params = layer.params
    if areas.index(layer.target_name) > areas.index(layer.source_name):
        connections[layer_name] = FFConnection(layer_name, params.in_channels, params.out_channels,\
                    params.kernel_size, params.gsh, params.gsw, INPUT_SIZE[1],stride=params.stride, \
                    mask=mask, padding=params.padding,padding_mode = params.padding_mode)
            



mn.calc_graph['LGNd'] = mn.LGNd(input,mn.calc_graph['LGNd'])
for region in mn.regions:
    for t,target in enumerate(region.targets):
        mn.calc_graph[target] = mn.connections[region.convs[t]](mn.calc_graph[region.name],mn.calc_graph[target])
out=torch.cat([torch.flatten( torch.nn.AdaptiveAvgPool2d(4)(mn.calc_graph[area]),1) for area in OUTPUT_AREAS],axis=1)
out = mn.classifier(out)



n_steps = 1
for n in range(n_steps):
    mn.calc_graph['LGNd'] = mn.LGNd(data,mn.calc_graph['LGNd'])
    for region in mn.regions:
        for t,target in enumerate(region.targets):
            mn.calc_graph[target] = mn.connections[region.convs[t]](mn.calc_graph[region.name],mn.calc_graph[target])
out=torch.cat([torch.flatten( torch.nn.AdaptiveAvgPool2d(4)(mn.calc_graph[area]),1) for area in OUTPUT_AREAS],axis=1)
out = mn.classifier(out)
       




Convs = torch.nn.ModuleDict()
G, _ = net.make_graph(recurrent = recurrent)
 
if not recurrent:
    areas = list(nx.topological_sort(G))
else:
    areas = net.hierarchical_order
     
            
        
        
layer = net.find_conv_source_target('input', 'LGNd')
params = layer.params
layer_name = layer_name = "_".join((layer.source_name,layer.target_name))
Convs['LGNd'] = FFConnection(layer_name,params.in_channels, params.out_channels,\
                            params.kernel_size, params.gsh, params.gsw,INPUT_SIZE[1],stride=params.stride, \
                                mask=mask, padding=params.padding,padding_mode = params.padding_mode)



connections = nn.ModuleDict()
resolutions = {}
for layer in net.layers[1:]:
    layer_name = layer_name = "_".join((layer.source_name,layer.target_name))
    params = layer.params
    if areas.index(layer.target_name) > areas.index(layer.source_name):
        connections[layer_name] = FFConnection(layer_name, params.in_channels, params.out_channels,\
                    params.kernel_size, params.gsh, params.gsw, INPUT_SIZE[1],stride=params.stride, \
                    mask=mask, padding=params.padding,padding_mode = params.padding_mode)    
        
#regions = [Region(region,net,areas) for region  in areas[1:]]




       


for region in areas[1:]:
    RConvs = torch.nn.ModuleDict()
    source_layers = [layer for layer in net.layers if layer.target_name == region]  
    for sl in source_layers:
           params = sl.params
           print('layer.source_name',sl.source_name, 'layer.target_name',sl.target_name )
           if areas.index(sl.target_name) > areas.index(sl.source_name):
               RConvs[sl.source_name] = FFConnection(params.in_channels, params.out_channels,\
                               params.kernel_size, params.gsh, params.gsw, INPUT_SIZE[1],stride=params.stride, \
                                   mask=mask, padding=params.padding)
           else:
               RConvs[sl.source_name ] = RecurrentConnection(params.in_channels, params.out_channels,\
                               params.kernel_size, params.gsh, params.gsw, INPUT_SIZE[1],stride=params.stride, \
                                  mask=mask, padding=params.padding)
                   
'''