#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 19:04:11 2022

@author: danamastrovito
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "cmouse"), str(REPO / "cmouse" / "exps" / "cifar")]
from cifar_config import *
from train_config import *
import torch
import torch.optim as optim
import network
import mousenet_model
from numpy.random import default_rng
import random
import numpy as np
from mousenet_model import *
import matplotlib.pyplot as plt
import pandas as pd
import glob

import torchvision
import torchvision.transforms as transforms


def debug_memory():
    if device.type == 'cuda':
        mousenet_model.debug_memory()


def get_data_loaders(Grayscale=False):
    if INPUT_SIZE[0] != 3:
        raise ValueError('INPUT_SIZE must have 3 channels')
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    resize = transforms.Resize(INPUT_SIZE[1:], interpolation=transforms.InterpolationMode.BICUBIC)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                          resize, transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([resize, transforms.ToTensor(), normalize])
    if Grayscale:
        transform_train.transforms.append(transforms.Grayscale(3))
        transform_test.transforms.append(transforms.Grayscale(3))
    datasets = {'cifar10': torchvision.datasets.CIFAR10, 'cifar100': torchvision.datasets.CIFAR100}
    if DATASET not in datasets:
        raise ValueError('DATASET should be cifar10 or cifar100')
    trainset = datasets[DATASET](root=DATA_DIR, train=True, download=True, transform=transform_train)
    testset = datasets[DATASET](root=DATA_DIR, train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
    return train_loader, test_loader


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience and metrics >= 90:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


def plot_states(states,outdir,suffix=""):
    
    #plt.clf()
    for r,region in enumerate(list(states.keys())):
        plt.clf()
        #plt.subplot(4,6,r+1)
        plt.plot(np.arange(1,len(states[region])+1),np.array(states[region]),marker='.')
        plt.title(region)
        plt.savefig(os.path.join(outdir,"_".join((region.replace("/","_"),'states_over_time_'+suffix+'.png'))))
    #plt.tight_layout()
    #plt.savefig("states_over_time.png",dpi = 300)
    
    
def plot_loss(loss,outdir, accuracy = None, validation = False,acc = None,suffix = ""):
    plt.clf()
    if validation:
        plt.subplot(2,1,1)
        plt.plot(loss)
        plt.title("CE Loss")
        plt.subplot(2,1,2)
        plt.plot(accuracy)
        if acc is not None:
            plt.title( acc +" %")
        else:
            plt.title("accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,"Validation_loss_"+suffix+".png"))
    else:
        plt.plot(loss)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "Training_loss_"+suffix+".png"))

           
        
 
       
#file = os.path.join(basedir,'cmouse/exps/cifar/myresults',"mask_3_cifar10_LR_0.001_M_0.5_mousenet/42_83.62.pt")
#file = os.path.join(basedir,'cmouse/exps/cifar/myresults/recurrent/sampled',"mask_3_cifar10_LR_0.001_M_0.5_mousenet/42_10.0.pt")
#file = os.path.join(basedir,'cmouse/exps/cifar/myresults/recurrent/sampled/max_tanh_multiplicative_recurrence_3_cifar10_LR_0.005_M_0.5_mousenet/42_57.72.pt')
#file = os.path.join(basedir,"cmouse/exps/cifar/myresults/recurrent/baseline/sigmoid_multiplicative_recurrence_3_cifar10_LR_0.05_M_0.5_mousenet/42_init.pt")
#file = os.path.join(basedir,'cmouse/exps/cifar/myresults/recurrent/sampled/ReLU_eachstep_multiplicative_recurrence_3_cifar10_LR_0.005_M_0.5_mousenet/42_37.95.pt')

GRAY = False
if GRAY:
    outfile_suffix = "_gray"
else:
    outfile_suffix =""
    
parser = argparse.ArgumentParser(description='Train MouseNet on CIFAR.')
parser.add_argument('--mode', choices=['recurrent', 'feedforward'], default='recurrent')
args = parser.parse_args()
recurrent = args.mode == 'recurrent'
modelfile = ('recurrent_' if recurrent else '') + 'mousenet_inputsize64_ccf_2017'


RUN = 1
suffix = "_run_"+str(RUN)
outfile_suffix = "_SGD_scheduledLR_BN2d_relu"+outfile_suffix+suffix
#outfile_suffix = "Adam_BN2d"+outfile_suffix+suffix

print(modelfile, outfile_suffix)

SEED = 42 + RUN
rng = default_rng(SEED)
torch.backends.cudnn.deterministic = True
random.seed(SEED)       # python random seed
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED)    # numpy random seed

step_range= (16,20)

mask = 3
#net = network.load_network_from_pickle(os.path.join(basedir,'network_complete_updated_number(3,64,64)_edited_sigma_recurrent.pkl'))
net = network.load_network_from_pickle(str(REPO / "models" / (modelfile + ".pkl")))
outdir = os.path.join(RESULT_DIR, modelfile + outfile_suffix)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mn= mousenet_model.mousenet(net, recurrent = recurrent,device=device,mask=mask)



os.makedirs(outdir, exist_ok=True)
list_of_files = glob.glob(os.path.join(outdir,'*.pth'))
lr = LR
if not list_of_files:
    start = 0
else:
    latest_file = max(list_of_files, key=os.path.getctime)
    epoch = int(os.path.basename(latest_file).split('epoch')[1].split("_")[0])
    scheduled_lr_reductions = int(epoch/10)
    lr = LR
    for i in range(scheduled_lr_reductions):
        lr = lr*.8
    print("Reloading model")
    chkpt = torch.load(latest_file, map_location=device)
    start = epoch
    mn.load_state_dict(chkpt)

    

train_loader, test_loader = get_data_loaders(Grayscale = GRAY)


'''
sparsity = {}
sparsity['connection_name'] = []
sparsity['connection_sparsity'] = []
sparsity['mean_weights'] = []
sparsity['mean_bias'] = []

with torch.no_grad():
    for region in mn.regions:
        for key in region.convs.keys():
            sparsity['connection_name'].append("_".join((key, region.name)))
            sparsity['connection_sparsity'].append(1-(region.convs[key].conv[1].mask.sum()/region.convs[key].conv[1].mask.numel()).cpu().numpy())
            sparsity['mean_weights'].append((region.convs[key].conv[1].mask*region.convs[key].conv[1].weight.data).mean().cpu().numpy())
            sparsity['mean_bias'].append(region.convs[key].conv[1].bias.data.mean().cpu().numpy())
            
            
df = pd.DataFrame(sparsity)
df.to_csv('recurrent_mousenet_sparsity.csv')

'''


loss = nn.CrossEntropyLoss()





mn.to(device)
params = list(mn.named_parameters())
#lr = 0.001
#optimizer = optim.Adam(mn.parameters(recurse = True),lr = lr,amsgrad = True)



optimizer = optim.SGD(mn.parameters(recurse = True),lr = lr,momentum = MOMENTUM)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.8)

#optimizer = optim.SGD(mn.parameters(recurse=True),lr = 0.1)

#mn.train()
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
accuracy = []
es = EarlyStopping(mode = 'max')
  
for epoch in range(start, EPOCHS + 1):
    mn.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        mn.reset(BATCH_SIZE, device)
        data, labels = data.to(device), labels.to(device)
        if recurrent:
            n_steps = rng.integers(low=step_range[0], high=step_range[1], size=1)[0]   
        else:
            n_steps = 6
        out = mn(data,n_steps = n_steps)
        l = loss(out, labels)
        l.backward()
        
        optimizer.step()
        Trainloss.append(l.detach().cpu().numpy())
        if batch_idx %10 ==0:
            debug_memory()
            #plot_states(states,outdir) ##########
            plot_loss(Trainloss,outdir) ############
            print(n_steps,epoch,l.detach().cpu().numpy(),batch_idx, "Of",len(train_loader))
        
        if batch_idx%100 ==0:
            torch.save(mn.state_dict(),os.path.join(outdir,modelfile+"_epoch" + str(epoch)+"_batch_idx"+str(batch_idx)+'.pth')) ####################
            #lr = lr*.1
            #optimizer = optim.Adam(mn.parameters(),lr = lr)
        
        '''
        if batch_idx %100 == 0:
            plt.clf()
            for i in range(net.layers):
                plot(states[i])
        '''
    with torch.no_grad():
        mn.eval()
        correct = 0
        for data, target in test_loader:
            # Load the input features and labels from the test dataset
            data, target = data.to(device), target.to(device)
            mn.reset(BATCH_SIZE,device)
            # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
            if recurrent:
                n_steps = rng.integers(low=step_range[0], high=step_range[1], size=1)[0]   
            else:
                n_steps = 6
            output = mn(data, n_steps=n_steps)                 
            
            # Compute the loss sum up batch loss
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            test_loss = loss(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()         
            Vloss.append(test_loss.cpu().numpy())
        acc = 100. * correct / len(test_loader.dataset)
        accuracy.append(acc.item())
        plot_loss(Vloss,outdir, accuracy= accuracy, validation = True,acc = " ".join(("epoch",str(epoch), "acc",str(acc))))
        print(acc)
        if es.step(acc):
            break
    scheduler.step()
    print(scheduler.get_lr())
      
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