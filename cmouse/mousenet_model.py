#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:11:16 2022

@author: danamastrovito
"""


import torch
from torch import nn
import networkx as nx
import numpy as np
from cifar_config import  INPUT_SIZE, EDGE_Z, OUTPUT_AREAS, HIDDEN_LINEAR, NUM_CLASSES
#import collections, gc, resource, torch OrderedDict
from collections import OrderedDict
import torchvision
import sys


def debug_memory():
    '''
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter(
        (str(o.device), str(o.dtype), tuple(o.shape))
        for o in gc.get_objects()
        if torch.is_tensor(o)
    )
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))
    '''
    
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))



def get_out_sigma(source_area, source_depth, target_area, target_depth,source_hierarchy,target_hierarchy):
    if target_depth == '4':
        if target_area != 'VISp' and target_area != 'VISpor' and target_hierarchy > source_hierarchy:
            return 1/2
        if target_area == 'VISpor':
            if source_area == 'VISp' and target_hierarchy > source_hierarchy:
                return 1/2 
    if source_depth == '4':
        if source_area != 'VISp' and source_area != 'VISpor' and source_hierarchy > target_hierarchy:
            return 2
        if source_area == 'VISpor':
            if target_area == 'VISp' and source_hierarchy > target_hierarchy:
                return 2        
    return 1


    
class Conv2dMask(nn.Conv2d):
    """
    Conv2d with Gaussian mask 
    """
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, mask=3, stride=1, padding=0,padding_mode= 'zeros'):
        super(Conv2dMask, self).__init__(in_channels, out_channels, kernel_size, stride=stride,bias= True)
        if not padding_mode == 'replicate':
            self.mypadding = nn.ConstantPad2d(padding, 0)
        else:
            self.mypadding = nn.ReflectionPad2d(padding)
        if mask == 0:
            self.mask = None
        if mask==1:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)))
        elif mask ==2:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)), requires_grad=False) 
        elif mask ==3:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask_vary_channel(gsh, gsw, kernel_size, out_channels, in_channels)), requires_grad=False)
        else:
            assert("mask should be 0, 1, 2, 3!")
            
        if self.mask is not None:
            with torch.no_grad():
                self.weight.data /= self.mask.mean((1,2,3), keepdims=True).sqrt()
                self.weight.data = torch.nan_to_num(self.weight.data)
          
    def forward(self, input):
        if self.mask is not None:
            return super(Conv2dMask, self)._conv_forward(self.mypadding(input), self.weight*self.mask, self.bias)
        else:
            return super(Conv2dMask, self)._conv_forward(self.mypadding(input), self.weight, self.bias)  
            
    def make_gaussian_kernel_mask(self, peak, sigma):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        width = int(sigma*EDGE_Z)        
        x = np.arange(-width, width+1)
        X, Y = np.meshgrid(x, x)
        radius = np.sqrt(X**2 + Y**2)

        probability = peak * np.exp(-radius**2/2/sigma**2)

        re = np.random.rand(len(x), len(x)) < probability
        # plt.imshow(re, cmap='Greys')
        return re
    
    def make_gaussian_kernel_mask_vary_channel(self, peak, sigma, kernel_size, out_channels, in_channels):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :param kernel_size: kernel size of the conv2d 
        :param out_channels: number of output channels of the conv2d
        :param in_channels: number of input channels of the con2d
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        re = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        for i in range(out_channels):
            for j in range(in_channels):
                re[i, j, :] = self.make_gaussian_kernel_mask(peak, sigma)
        return re




class ConvTranspose2dMask(nn.ConvTranspose2d):
    """
    Conv2d with Gaussian mask 
    """
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, mask=3, stride=1,dilation=1, output_padding=0,padding = 0):
        super(ConvTranspose2dMask, self).__init__(in_channels, out_channels, kernel_size,stride= stride,padding = padding,output_padding = output_padding,dilation = dilation,bias = True)
        if mask == 0:
            self.mask = None
        if mask==1:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)))
        elif mask ==2:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)), requires_grad=False) 
        elif mask ==3:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask_vary_channel(gsh, gsw, kernel_size, in_channels, out_channels)), requires_grad=False)
        else:
            assert("mask should be 0, 1, 2, 3!")
        
        if self.mask is not None:
            with torch.no_grad():
                self.weight.data /= self.mask.mean((1,2,3), keepdims=True).sqrt()
                self.weight.data = torch.nan_to_num(self.weight.data)
                
    def forward(self, input):
        if self.mask is not None:
            self.weight.data = self.weight.data*self.mask
            return super(ConvTranspose2dMask, self).forward(input)
        else:
            return super(ConvTranspose2dMask, self).forward(input)
            
    def make_gaussian_kernel_mask(self, peak, sigma):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        width = int(sigma*EDGE_Z)        
        x = np.arange(-width, width+1)
        X, Y = np.meshgrid(x, x)
        radius = np.sqrt(X**2 + Y**2)

        probability = peak * np.exp(-radius**2/2/sigma**2)

        re = np.random.rand(len(x), len(x)) < probability
        # plt.imshow(re, cmap='Greys')
        return re
    
    def make_gaussian_kernel_mask_vary_channel(self, peak, sigma, kernel_size, out_channels, in_channels):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :param kernel_size: kernel size of the conv2d 
        :param out_channels: number of output channels of the conv2d
        :param in_channels: number of input channels of the con2d
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        re = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        for i in range(out_channels):
            for j in range(in_channels):
                re[i, j, :] = self.make_gaussian_kernel_mask(peak, sigma)
        return re


class FFConnection(nn.Module):
    def __init__(self, name,in_channels, out_channels, kernel_size, gsh, gsw, in_resolution,out_resolution,mask=3, stride=1, padding=0,padding_mode= 'zeros'):
        super(FFConnection, self).__init__()
        self.name = name
        self.conv = nn.Sequential(nn.ReLU(), \
                    Conv2dMask(in_channels, out_channels, kernel_size, gsh, gsw, stride=stride,mask=mask, padding=padding,padding_mode= padding_mode)) 
                 
    def forward(self,x,state=None):
        out =  self.conv(x)
        return out 
  
class RecurrentConnection(nn.Module):
    def __init__(self, name,in_channels, out_channels, kernel_size, gsh, gsw, in_resolution,out_resolution, stride=1, dilation = 1,output_padding=0,mask=3,padding = 0):
        super(RecurrentConnection, self).__init__()
        self.name = name
        self.conv = nn.Sequential(nn.ReLU(), \
            ConvTranspose2dMask(in_channels, out_channels, kernel_size, gsh, gsw, mask=mask,stride=stride,dilation=dilation,output_padding = output_padding,padding = padding)) 
        
    def forward(self,x,state):
        out = state*self.conv(x)
        return out
       
   
class Region(nn.Module):
    def __init__(self,region,sources,convolutional_layers,params,device = 'cpu'):
        super(Region,self).__init__()
        self.name = region
        self.sources =  sources
        self.convs = convolutional_layers
        self.out_channels = params['out_channels']
        self.out_size = int(params['out_size'])
        self.decay = nn.Parameter(torch.zeros((1,self.out_channels, self.out_size, self.out_size),device=device))
        self.init_state = nn.Parameter(torch.zeros((self.out_channels,self.out_size, self.out_size),device=device)+1e-3)
    
    def forward(self,states):
        tanh = nn.Tanh()
        state = torch.stack([self.convs[source](states[source],self.state) for source in self.sources])
        #self.state = self.state*self.decay.sigmoid() + state.sum(dim=0)
        self.state = tanh(self.state + state.sum(dim=0))
        
         
    def reset(self,batch_size,device):
        self.state = self.init_state.unsqueeze(0).repeat(batch_size, 1, 1, 1).clone()
        
    
class LGNd(nn.Module):
    def __init__(self,convolutional_layers,params):
       super(LGNd,self).__init__() 
       self.out_channels = params.out_channels
       self.out_size = int(params.out_size)
       self.convs = convolutional_layers
       
    def reset(self, batch_size,device):
        self.state = torch.zeros((batch_size,self.out_channels, self.out_size,self.out_size),requires_grad = True,device=device)
       
    def forward(self,input):
        self.state = self.convs(input)
        return self.state
        
class mousenet(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=3,recurrent = False,device = 'cpu'):
        super(mousenet, self).__init__()
        network = network
        
        G, _ = network.make_graph(recurrent = recurrent)
        
        if not recurrent:
            areas = list(nx.topological_sort(G))
        else:
            areas = network.hierarchical_order
        
        layer = network.find_conv_source_target('input', 'LGNd')
        params = layer.params
        layer_name = layer_name = "_".join((layer.source_name,layer.target_name))
        print(layer_name)
        self.LGNd = LGNd(FFConnection(layer_name, params.in_channels, params.out_channels,\
                                    params.kernel_size, params.gsh, params.gsw, params.in_size, params.out_size,stride=params.stride, \
                                        mask=mask, padding=params.padding,padding_mode = params.padding_mode),params)
        #self.BN = nn.BatchNorm2d(params.in_channels) 
        self.connections = nn.ModuleDict()
        region_params = {}
        for layer in network.layers[1:]:
            layer_name = layer_name = "_".join((layer.source_name,layer.target_name))
            print(layer_name)
            params = layer.params
            if layer.target_name not in region_params.keys():
                region_params[layer.target_name] = {'out_channels':params.out_channels,'out_size':params.out_size}
            if areas.index(layer.target_name) > areas.index(layer.source_name):
                self.connections[layer_name] = FFConnection(layer_name, params.in_channels, params.out_channels,\
                            params.kernel_size, params.gsh, params.gsw, params.in_size,params.out_size,stride=params.stride, \
                            mask=mask, padding=params.padding,padding_mode = params.padding_mode)
            else:
                if type(params.output_padding) == tuple:
                    params.output_padding = tuple(int(item) for item in params.output_padding)
                else:
                    params.output_padding = int(params.output_padding)
                self.connections[layer_name] = RecurrentConnection(layer_name, params.in_channels, params.out_channels,\
                            params.kernel_size, params.gsh, params.gsw, params.in_size,params.out_size,stride = params.stride, \
                            dilation = params.dilation,output_padding = params.output_padding,mask=mask,padding = params.padding)
        
        
        self.regions = torch.nn.ModuleList()
        for area in areas[1:]:
            layers = [ layer for layer in network.layers if layer.target_name == area]
            sources = [layer.source_name for layer in layers]
            convs = {}
            for layer in layers:
            #if "_".join((layer.source_name,region)) in self.connections.keys():
            #sources.append(layer.source_name)
                convs[layer.source_name] = self.connections["_".join((layer.source_name,area))]
            self.regions.append(Region(area,sources,convs,region_params[area],device=device))
      
        total_size=0
        
        for area in OUTPUT_AREAS:
            layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
            total_size += int(16*layer.params.out_channels)
           
        self.classifier = nn.Linear(int(total_size), NUM_CLASSES)
            
    
        
    def reset(self,batch_size,device):
        self.LGNd.reset(batch_size,device)
        self.calc_graph = {}
        self.calc_graph['LGNd'] = self.LGNd.state
        for region in self.regions:
            region.reset(batch_size,device)
            self.calc_graph[region.name] = region.state.clone()
    
    def step(self):
       for region in self.regions:
           region(self.calc_graph)
       for region in self.regions:
           self.calc_graph[region.name] = region.state.clone()
           
    def forward(self, input,n_steps = 10,return_calc_graph = False,track_states = False):        
        states = {}
        if track_states:
            for region in self.regions:
                states[region.name] = []
        
        if n_steps > 6:
            no_grad_steps = n_steps - 6
            #input = self.BN(input)
            with torch.no_grad():
                self.calc_graph['LGNd'] = self.LGNd(input)
                for n in range(no_grad_steps):
                    self.step()
                    if track_states:
                        for region in self.regions:
                            states[region.name].append(self.calc_graph[region.name][1].mean().clone().detach().cpu().numpy())#mean for a single batch
        
        self.calc_graph['LGNd'] = self.LGNd(input)
        for n in range(6):
            self.step()
            if track_states:
                with torch.no_grad():
                    for region in self.regions:
                        states[region.name].append(self.calc_graph[region.name][1].mean().clone().detach().cpu().numpy())
                
        out=torch.cat([torch.flatten( torch.nn.AdaptiveAvgPool2d(4)(self.calc_graph[area]),1) for area in OUTPUT_AREAS],axis=1)
        out = self.classifier(out)
        return out,states
    
    
  