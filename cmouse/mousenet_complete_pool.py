import torch
from torch import nn
import networkx as nx
import numpy as np
from config import  INPUT_SIZE, EDGE_Z, OUTPUT_AREAS, HIDDEN_LINEAR, NUM_CLASSES
import collections, gc, resource, torch

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


class Conv2dMask(nn.Conv2d):
    """
    Conv2d with Gaussian mask 
    """
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, mask=3, stride=1, padding=0,padding_mode= 'zeros'):
        super(Conv2dMask, self).__init__(in_channels, out_channels, kernel_size, stride=stride)
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

class MouseNetCompletePool(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=3,recurrent = False):
        super(MouseNetCompletePool, self).__init__()
        self.Convs = nn.ModuleDict()
        self.BNs = nn.ModuleDict()
        self.network = network
        
        G, _ = network.make_graph(recurrent = recurrent)
        
        if not recurrent:
            self.areas = list(nx.topological_sort(G))
            padding_mode = 'zeros'
        else:
            self.areas = network.hierarchical_order
            
        
        for layer in network.layers:
            params = layer.params
            print('layer.source_name',layer.source_name, 'layer.target_name',layer.target_name )
            self.Convs[layer.source_name + layer.target_name] = Conv2dMask(params.in_channels, params.out_channels, params.kernel_size,
                                                    params.gsh, params.gsw, stride=params.stride, mask=mask, padding=params.padding,padding_mode = params.padding_mode)
            ## plotting Gaussian mask
            #plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
            #plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))
            if layer.target_name not in self.BNs:
                self.BNs[layer.target_name] = nn.BatchNorm2d(params.out_channels)

        # calculate total size output to classifier
        total_size=0
        
        for area in OUTPUT_AREAS:
            layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
            total_size += int(16*layer.params.out_channels)
        #     if area =='VISp5':
        #         layer = network.find_conv_source_target('VISp2/3','VISp5')
        #         visp_out = layer.params.out_channels
        #         # create 1x1 Conv downsampler for VISp5
        #         visp_downsample_channels = visp_out
        #         ds_stride = 2
        #         self.visp5_downsampler = nn.Conv2d(visp_out, visp_downsample_channels, 1, stride=ds_stride)
        #         total_size += INPUT_SIZE[1]/ds_stride * INPUT_SIZE[2]/ds_stride * visp_downsample_channels
        #     else:
        #         layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
        #         total_size += int(layer.out_size*layer.out_size*layer.params.out_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(int(total_size), NUM_CLASSES),
            # nn.Linear(int(total_size), HIDDEN_LINEAR),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(HIDDEN_LINEAR, HIDDEN_LINEAR),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(HIDDEN_LINEAR, NUM_CLASSES),
        )
                     
    
    def get_img_feature_recurrent_slow(self, x, area_list, flatten=True,n_steps = None,return_calc_graph = False):
        nt = self.network.hierarchy_depth
        calc_graph = {}
        #size_mismatch = []
        area = 'LGNd'
        layer = self.network.find_conv_source_target('input', area)
        layer_name = layer.source_name + layer.target_name
        calc_graph[area] =  nn.ReLU(inplace=True)(self.BNs[area](self.Convs[layer_name](x)))
        source_areas = ['LGNd']
        #edges_computed = []
        
        finished = False
        ncalc = 0 
        nt = 0
        #ec = []
        
        
        while len(source_areas) >0 and not finished:
            #print(source_areas)
            if len(source_areas) == len(self.network.layers) - 2:
                if n_steps is not None:
                    if nt == n_steps:
                        finished = True
                else:
                    finished = True
            target_layers = [layer for layer in self.network.layers if layer.source_name in source_areas]
            targets = [layer.target_name for layer in target_layers]
            for layer in target_layers:
                layer_name = layer.source_name + layer.target_name
                convolution = self.Convs[layer_name](calc_graph[layer.source_name])
                if layer.target_name not in calc_graph:
                    calc_graph[layer.target_name] = convolution
                    ncalc += 1
                    #ec.append(layer_name)
                #elif layer.out_size != convolution.shape[2]:
                #    pad = nn.ConstantPad2d(int((layer.out_size-calc_graph[layer.source_name].shape[2])/2),0)
                #    calc_graph[layer.target_name] = calc_graph[layer.target_name] + self.Convs[layer_name](pad(calc_graph[layer.source_name]))
                    
                else:
                      calc_graph[layer.target_name] = calc_graph[layer.target_name] + convolution
                      ncalc += 1
                      #ec.append(layer_name)
                      
            for target in targets:
                calc_graph[target] = torch.nn.ReLU(inplace=True)(self.BNs[target](calc_graph[target]))
                
            #target_layers = [layer for layer in self.network.layers if layer.source_name in targets]
            source_areas = targets
            #edges_computed.append(ec)
            #ec = []
            nt += 1

        #print(ncalc,nt)
        if len(area_list) == 1:
            if flatten:
                return torch.flatten(calc_graph['%s'%(area_list[0])], 1)
            else:
                return calc_graph['%s'%(area_list[0])]
        else:
            re = None
            for area in area_list:
                if re is None:
                    re = torch.flatten(torch.nn.AdaptiveAvgPool2d(4) (calc_graph[area]), 1)
                    # re = torch.flatten(
                        # nn.ReLU(inplace=True)(self.BNs['%s_downsample'%area](self.Convs['%s_downsample'%area](calc_graph[area]))), 
                        # 1)
                else:
                    re=torch.cat([torch.flatten(    
                        torch.nn.AdaptiveAvgPool2d(4) (calc_graph[area]),1), re], axis=1)
            if return_calc_graph:
                return re,calc_graph
            else:
                del calc_graph
                return re, None
                        
              
    def get_img_feature(self, x, area_list, flatten=True,SUBFIELDS=False,return_calc_graph = False):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :param area_list: a list of area names
        :return: if list length is 1, return the (flatten/unflatten) activation of that area
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        calc_graph = {}

        for area in self.areas:
            if area == 'input':
                continue
   
            if area == 'LGNd' or area == 'LGNv':
                layer = self.network.find_conv_source_target('input', area)
                layer_name = layer.source_name + layer.target_name

                if SUBFIELDS:
                    left, width, bottom, height = self.sub_indices[layer_name]
                    source_field = torch.narrow(torch.narrow(x, 2, left, width), 3, bottom, height) #TODO: check top/bottom direction
                    calc_graph[area] =  nn.ReLU(inplace=True)(self.BNs[area](self.Convs[layer_name](source_field)))
                else:
                    calc_graph[area] =  nn.ReLU(inplace=True)(self.BNs[area](self.Convs[layer_name](x)))

                continue

            for layer in self.network.layers:
                if layer.target_name == area:
                    layer_name = layer.source_name + layer.target_name

                    if SUBFIELDS:
                        left, width, bottom, height = self.sub_indices[layer_name] #TODO: incorporate padding here
                        source_field = torch.narrow(torch.narrow(calc_graph[layer.source_name], 2, left, width), 3, bottom, height)
                        layer_output = self.Convs[layer_name](source_field)
                    else:
                        layer_output = self.Convs[layer_name](calc_graph[layer.source_name])

                    if area not in calc_graph:
                        calc_graph[area] = layer_output
                    else:
                        calc_graph[area] = calc_graph[area] + layer_output
            calc_graph[area] = nn.ReLU(inplace=True)(self.BNs[area](calc_graph[area]))
        
        if len(area_list) == 1:
            if flatten:
                return torch.flatten(calc_graph['%s'%(area_list[0])], 1)
            else:
                return calc_graph['%s'%(area_list[0])]

        else:
            re = None
            for area in area_list:
                if re is None:
                    re = torch.flatten(torch.nn.AdaptiveAvgPool2d(4) (calc_graph[area]), 1)
                    # re = torch.flatten(
                        # nn.ReLU(inplace=True)(self.BNs['%s_downsample'%area](self.Convs['%s_downsample'%area](calc_graph[area]))), 
                        # 1)
                else:
                    re=torch.cat([torch.flatten(    
                        torch.nn.AdaptiveAvgPool2d(4) (calc_graph[area]), 
                        1), re], axis=1)
                    # re=torch.cat([
                        # torch.flatten(
                        # nn.ReLU(inplace=True)(self.BNs['%s_downsample'%area](self.Convs['%s_downsample'%area](calc_graph[area]))), 
                        # 1), 
                        # re], axis=1)
                # if area == 'VISp5':
                #     re=torch.flatten(self.visp5_downsampler(calc_graph['VISp5']), 1)
                # else:
                #     if re is not None:
                #         re = torch.cat([torch.flatten(calc_graph[area], 1), re], axis=1)
                #     else:
                #         re = torch.flatten(calc_graph[area], 1)
        
        if return_calc_graph:
            return re,calc_graph
        else:
            del calc_graph
            return re, None
    
    '''
    def get_img_feature(self, x, area_list, flatten=True):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :param area_list: a list of area names
        :return: if list length is 1, return the (flatten/unflatten) activation of that area
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        calc_graph = {}
        size_mismatch =[]
        ncalc = 0
        edges_computed = []
        for area in self.areas:
            if area == 'input':
                continue
            print('area',area)
            if area == 'LGNd' or area == 'LGNv':
                layer = self.network.find_conv_source_target('input', area)
                layer_name = layer.source_name + layer.target_name
                calc_graph[area] =  nn.ReLU(inplace=True)(self.BNs[area](self.Convs[layer_name](x)))
                continue
            for layer in self.network.layers:
                if layer.target_name == area:
                    layer_name = layer.source_name + layer.target_name
                    print('layer_name',layer_name,'layer.source_name',layer.source_name)
                    if area not in calc_graph and layer.source_name in calc_graph:
                        calc_graph[area] = self.Convs[layer_name](calc_graph[layer.source_name])
                        ncalc += 1
                    elif area in calc_graph and layer.source_name in calc_graph:
                        try:
                            calc_graph[area] = calc_graph[area] + self.Convs[layer_name](calc_graph[layer.source_name]) 
                            ncalc += 1
                        except RuntimeError:
                                size_mismatch.append((layer_name,area, layer.source_name))
                                print('layer_name',layer_name,'layer.source_name',layer.source_name,self.Convs[layer_name])
            calc_graph[area] = nn.ReLU(inplace=True)(self.BNs[area](calc_graph[area]))
        
        print(ncalc)
        if len(area_list) == 1:
            if flatten:
                return torch.flatten(calc_graph['%s'%(area_list[0])], 1)
            else:
                return calc_graph['%s'%(area_list[0])]

        else:
            re = None
            for area in area_list:
                if re is None:
                    re = torch.flatten(torch.nn.AdaptiveAvgPool2d(4) (calc_graph[area]), 1)
                    # re = torch.flatten(
                        # nn.ReLU(inplace=True)(self.BNs['%s_downsample'%area](self.Convs['%s_downsample'%area](calc_graph[area]))), 
                        # 1)
                else:
                    re=torch.cat([torch.flatten(    
                        torch.nn.AdaptiveAvgPool2d(4) (calc_graph[area]), 
                        1), re], axis=1)
                    # re=torch.cat([
                        # torch.flatten(
                        # nn.ReLU(inplace=True)(self.BNs['%s_downsample'%area](self.Convs['%s_downsample'%area](calc_graph[area]))), 
                        # 1), 
                        # re], axis=1)
                # if area == 'VISp5':
                #     re=torch.flatten(self.visp5_downsampler(calc_graph['VISp5']), 1)
                # else:
                #     if re is not None:
                #         re = torch.cat([torch.flatten(calc_graph[area], 1), re], axis=1)
                #     else:
                #         re = torch.flatten(calc_graph[area], 1)
            return re,size_mismatch
    '''
    
    def forward(self, x,n_steps=None,return_calc_graph = False):
        if self.network.recurrent:
            x,calc_graph = self.get_img_feature_recurrent_slow(x,OUTPUT_AREAS,n_steps=n_steps,return_calc_graph = return_calc_graph)
        else:
            x,calc_graph = self.get_img_feature(x, OUTPUT_AREAS,return_calc_graph = return_calc_graph)
            
            
        #x = self.classifier(x)
        return x, calc_graph
