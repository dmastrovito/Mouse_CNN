import numpy as np
import networkx as nx
from anatomy import gen_anatomy
import torch
from torch import nn
from config import INPUT_SIZE, EDGE_Z, INPUT_GSH, INPUT_GSW, get_out_sigma
import os
import pickle
import matplotlib.pyplot as plt

class ConvParam:
    def __init__(self, in_channels, out_channels, gsh, gsw, out_sigma,in_size,out_size,FF = True):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param gsh: Gaussian height for generating Gaussian mask 
        :param gsw: Gaussian width for generating Gaussian mask
        :param out_sigma: ratio between output size and input size, 1/2 means reduce output size to 1/2 of the input size
        """

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gsh = gsh
        self.gsw = gsw
        self.kernel_size = 2*int(self.gsw * EDGE_Z) + 1
        self.in_size = in_size
        self.out_size = out_size
        
        self.padding_mode = "zeros"
        KmS = int((self.kernel_size-1/out_sigma))
        
        self.stride = int(1/out_sigma)
        if self.stride <= 0:
            self.stride = 1  
                           
        if not FF: #For recurrent connections using Transposed2dConv only
            '''
            ## for use with Conv2dassumes dilation = 1
            padding = int((self.kernel_size - self.in_size + self.out_size - self.stride)/2)
            padding_mode = "replicate"
            '''
            # for use iwth Conv2d Transpose assuming out_padding  = 0 and dilation =1
            self.output_padding = 0
            if self.out_size == self.in_size:
                self.stride = 1
                self.dilation = 1
                self.padding = int((self.kernel_size -1)/2)
                if not (self.padding % 1 ==0):
                    self.padding = int(np.floor(self.padding))
                    self.output_padding = int(self.out_size-self.in_size +2*self.padding -self.kernel_size +1)
            elif ((self.out_size - self.kernel_size) % (self.in_size -1)) == 0:
                self.stride = int((self.out_size - self.kernel_size)/(self.in_size -1))
                self.dilation = 1
                self.padding = 0
            else:
                self.stride = int(np.floor((self.out_size - self.kernel_size)/(self.in_size -1)))
                if self.stride <= 0:
                    self.stride = 1 
                if self.kernel_size != 1:
                    self.dilation =  int((self.out_size - self.stride * self.in_size - self.stride + 1 )/(self.kernel_size -1))
                    if self.dilation ==0:
                        self.dilation = 1
                        self.padding = (self.in_size - self.out_size +self.kernel_size -1)/2
                        if not (self.padding % 1 ==0):
                            self.padding = int(np.floor(self.padding))
                            self.output_padding = int(self.out_size-self.in_size +2*self.padding -self.kernel_size +1)
                        else:
                            self.padding = int(self.padding)
                    else:
                        lhs = self.out_size - self.in_size*self.stride +self.stride - self.dilation *(self.kernel_size -1) -1
                        if lhs %2 ==0 and lhs <=2: #even number
                            self.padding = int(lhs/2)
                        else:
                            self.output_padding = int(lhs)
                            self.padding = 0
                else:# kernel  == 1
                    self.dilation = 1
                    lhs = self.out_size - self.in_size*self.stride +self.stride - self.kernel_size
                    if lhs %2 ==0 and lhs <=2: #even number
                        self.padding = int(lhs/2)
                    else:
                        self.padding = 0
                        self.output_padding = int(lhs)
            if self.output_padding > self.stride  and self.output_padding > self.dilation:
                self.output_padding = 0
                self.padding = 0
                lhs = self.in_size*self.stride - self.out_size + 1 - self.stride + \
                    self.dilation*self.kernel_size - self.dilation
                while lhs < 0:
                    self.dilation = self.dilation + 1
                    lhs = self.in_size*self.stride - self.out_size + 1 - self.stride + \
                        self.dilation*self.kernel_size - self.dilation
                if lhs %2 ==0:
                    self.padding = int(lhs/2)
                else:
                    self.padding = int(lhs/2)
                    self.output_padding = int(lhs - self.padding)
                    
                
                    
                    
                    
                
                
                    
                
                
                '''    
                else:
                        self.output_padding = 0
                else:
                    self.dilation  = 1
                    self.output_padding = int(self.out_size-1 - self.in_size*self.stride + self.stride)
                '''        
            
        elif np.mod(KmS,2) == 0: #padding for FF
            self.padding = int(KmS/2)
        else: #alternative padding for FF
            self.padding = (int(KmS/2), int(KmS/2+1), int(KmS/2), int(KmS/2+1))
        
        
        
        
        
class ConvLayer:
    def __init__(self, source_name, target_name, params):
        """
        :param params: ConvParam containing the parameters of the layer
        :param source_name: name of the source area, e.g. VISp4, VISp2/3, VISp5
        :param target_name: name of the target area
        :param out_size: output size of the layer
        """
        self.params = params
        self.source_name = source_name
        self.target_name = target_name
        


class Network:
    """
    network class that contains all conv paramters needed to construct torch model.
    """
    def __init__(self):
        self.layers = []
        self.area_channels = {}
        self.area_size = {}
        
    def find_conv_source_target(self, source_name, target_name):
        for layer in self.layers:
            if layer.source_name == source_name and layer.target_name == target_name:
                return layer
        assert('no conv layer found!')
    
    def find_conv_target_area(self, target_name):
        for layer in self.layers:
            if layer.target_name == target_name:
                return layer
        assert('no conv layer found!')
         
        
    def hierarchical_sort(self, nodes,architecture):
        Layers = ['4','2/3','5']
        hierarchical_layer = {'0':0,'4':1,'2/3':2,'5':3}
        areas = []
        for node in nodes:
            lrep = [layer for layer in Layers if layer in node]
            if len(lrep) == 1:
                area = node.replace(lrep[0],"")
                areas.append((area,architecture.get_hierarchical_level(area),lrep[0]))
            elif len(lrep) ==0:
                area = node
                areas.append((area,architecture.get_hierarchical_level(area),'0'))
                
        areas_sorted = sorted(areas,key= lambda hierarchical: (hierarchical[1],hierarchical_layer[hierarchical[2]]))
        areas_sorted = [area[0]+area[2].replace('0','') for area in areas_sorted ]
        hierarchical_order = [areas_sorted.index(area) for area in nodes]
        return areas_sorted, hierarchical_order
        
    def hierarchical_edge_sort(self, edges,architecture):
        sources = [edge[0] for edge in edges]
        edge_order = []
        edges_sorted = []
        sources = list(set(sources))
        sources_sorted,sources_order = self.hierarchical_sort(sources, architecture)
        
        for source in sources_sorted:
            edges_w_source = [edge for e,edge in enumerate(edges) if source == edge[0] and \
                              self.hierarchical_order.index(edge[1]) > self.hierarchical_order.index(edge[0]) ]
            sorted_edge_targets,target_order  =  self.hierarchical_sort([edge[1] for edge in edges_w_source],architecture)
            for t,target in enumerate(sorted_edge_targets):
                if not (source,target) in edges_sorted:
                    edge_order.append(edges.index((source,target)))
                    edges_sorted.append((source,target))
            edges_w_source_recurrent = [edge for e,edge in enumerate(edges) if source == edge[0] and \
                              self.hierarchical_order.index(edge[1]) < self.hierarchical_order.index(edge[0]) ]
            sorted_edge_targets,target_order  =  self.hierarchical_sort([edge[1] for edge in edges_w_source_recurrent],architecture)
            for t,target in enumerate(sorted_edge_targets):
                if not (source,target) in edges_sorted:
                    edge_order.append(edges.index((source,target)))
                    edges_sorted.append((source,target))
            
        return edges_sorted, edge_order
               
            
        
        '''
        reverse = self.hierarchical_order.copy()
        reverse.reverse()
        
        edges_sorted = sorted(edges,key=lambda hierarchical:(self.hierarchical_order.index(hierarchical[0]),\
                                                             reverse.index(hierarchical[1])))
        edges_order = [edges.index(edge) for edge in edges_sorted]
        '''
  
        
    
    def construct_from_anatomy(self, anet, architecture,recurrent = False):
        """
        construct network from anatomy 
        :param anet: anatomy class which contains anatomical connections
        :param architecture: architecture class which calls set_num_channels for calculating connection strength
        """
        outchannels = []
        hierarchy_depth = 0
        # construct conv layer for input -> LGNd
        self.recurrent = recurrent
        
        self.area_channels['input'] = INPUT_SIZE[0]
        self.area_size['input'] = INPUT_SIZE[1]
        
        out_sigma = 1
        out_channels = np.floor(anet.find_layer('LGNd','').num/out_sigma/INPUT_SIZE[1]/INPUT_SIZE[2])
        architecture.set_num_channels('LGNd', '', out_channels)
        self.area_channels['LGNd'] = out_channels
        
        out_size =  INPUT_SIZE[1] * out_sigma
        self.area_size['LGNd'] = out_size
       
        convlayer = ConvLayer('input', 'LGNd',\
                              ConvParam(in_channels=INPUT_SIZE[0], \
                                        out_channels=out_channels,\
                                        gsh=INPUT_GSH,\
                                        gsw=INPUT_GSW, out_sigma=out_sigma,in_size = INPUT_SIZE,out_size = out_size))
        self.layers.append(convlayer)
       
        # construct conv layers for all other connections
        G, _ = anet.make_graph(recurrent =recurrent)
        Gtop = nx.topological_sort(G)
        root = next(Gtop) # get root of graph
        edge_list = list(G.edges)
        nodes = list(set([e[0].area+e[0].depth for e in edge_list]))
        self.hierarchical_order,_ = self.hierarchical_sort(nodes,architecture)
        if recurrent:
            _,edges_order = self.hierarchical_edge_sort([(e[0].area+e[0].depth,e[1].area+e[1].depth) for e in edge_list],architecture)
            edges = [edge_list[i] for i in edges_order]
        else:
            edges = nx.edge_bfs(G, root)
            
            
        for i, e in enumerate(edges):
            outchannels.append([])
            in_layer_name = e[0].area+e[0].depth
            out_layer_name = e[1].area+e[1].depth
            print('constructing layer %s: %s to %s'%(i, in_layer_name, out_layer_name))
            
            in_conv_layer = self.find_conv_target_area(in_layer_name)
            in_size = in_conv_layer.params.out_size
            in_channels = in_conv_layer.params.out_channels
            
            out_anat_layer = anet.find_layer(e[1].area, e[1].depth)
            level = architecture.get_hierarchical_level(e[0].area)
            hierarchy_depth = hierarchy_depth if hierarchy_depth  >  level else level
            if recurrent:
                out_sigma = get_out_sigma(e[0].area, e[0].depth, e[1].area, e[1].depth,\
                                     self.hierarchical_order.index(in_layer_name) ,self.hierarchical_order.index(out_layer_name))
                    
            else:
                out_sigma = get_out_sigma(e[0].area, e[0].depth, e[1].area, e[1].depth,\
                                     architecture.get_hierarchical_level(e[0].area) ,architecture.get_hierarchical_level(e[1].area))
            out_size = in_size * out_sigma
            
            
            self.area_size[e[1].area+e[1].depth] = out_size
            out_channels = np.floor(out_anat_layer.num/out_size**2)
            print(out_channels)
            outchannels[i].append(out_channels)
            architecture.set_num_channels(e[1].area, e[1].depth, out_channels)
            self.area_channels[e[1].area+e[1].depth] = out_channels
            
            if recurrent:
                if self.hierarchical_order.index(in_layer_name) < self.hierarchical_order.index(out_layer_name):
                    FF = True
                else:
                    FF = False
            else:
                FF = True
            
            convlayer = ConvLayer(in_layer_name, out_layer_name, 
                                  ConvParam(in_channels=in_channels, 
                                            out_channels=out_channels,
                                        gsh=architecture.get_kernel_peak_probability(e[0].area, e[0].depth, e[1].area, e[1].depth),
                                        gsw=architecture.get_kernel_width_pixels(e[0].area, e[0].depth, e[1].area, e[1].depth), \
                                        out_sigma=out_sigma,in_size = in_size,out_size= out_size,FF = FF))
            
                
                
            if self.hierarchical_order.index(in_layer_name) < self.hierarchical_order.index(out_layer_name):
                if type(convlayer.params.padding) == int:
                    conv_out_size = int(1 + ((in_size + (2*convlayer.params.padding) - 1*(convlayer.params.kernel_size -1) - 1)/convlayer.params.stride))
                else:
                    conv_out_size =int(1 + ((in_size + convlayer.params.padding[0] +convlayer.params.padding[1]  - 1*(convlayer.params.kernel_size -1) - 1)/convlayer.params.stride))
            else:
                conv_out_size = (in_size -1)*convlayer.params.stride -2*convlayer.params.padding + convlayer.params.dilation*(convlayer.params.kernel_size -1) + convlayer.params.output_padding + 1
            assert conv_out_size == out_size
                
            self.layers.append(convlayer)
        
        
        self.hierarchy_depth = hierarchy_depth
        
        return outchannels
            
         
       
            
    def make_graph(self,recurrent = False):
        """
        produce networkx graph
        """
        if recurrent:
           G = nx.MultiDiGraph() 
        else:
            G = nx.DiGraph()
        edges = [(p.source_name, p.target_name) for p in self.layers]
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        node_label_dict = { layer:'%s\n%s'%(layer, int(self.area_channels[layer])) for layer in G.nodes()}
        return G, node_label_dict

    def draw_graph(self, node_size=2000, node_color='yellow', edge_color='red',recurrent = False):
        """
        draw the network structure
        """
        G, node_label_dict = self.make_graph(recurrent = recurrent)
        edge_label_dict = {(c.source_name, c.target_name):(c.params.kernel_size) for c in self.layers}
        plt.figure(figsize=(12,12))
        if recurrent:
            if os.path.exists("FFmousenet_layout.pkl"):
                with open("FFmousenet_network_layout.pkl",'rb') as f:
                    pos = pickle.load(f)
        else:            
            pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color,alpha=0.4)
        nx.draw_networkx_labels(G, pos, node_label_dict, font_size=10,font_weight=640, alpha=0.7, font_color='black')
        nx.draw_networkx_edge_labels(G, pos, edge_label_dict, font_size=20, font_weight=640,alpha=0.7, font_color='red')
        if recurrent:
            plt.savefig("mousent_network_wrecurrence.png") 
        else:
            plt.savefig("mousent_network.png")  
        return pos


def gen_network_from_anatomy(architecture):
    anet = gen_anatomy(architecture)
    net = Network()
    net.construct_from_anatomy(anet, architecture)
    return net

def save_network_to_pickle(net, file_path):
    f = open(file_path,'wb')
    pickle.dump(net, f)

def load_network_from_pickle(file_path):
    f = open(file_path,'rb')
    net = pickle.load(f)
    return net

def gen_network(net_name, architecture, data_folder=''):
    file_path = './myresults/%s.pkl'%net_name
    if os.path.exists(file_path):
        net = load_network_from_pickle(file_path)
    else:
        net = gen_network_from_anatomy(architecture)
        if not os.path.exists('./myresults'):
            os.mkdir('./myresults')
        save_network_to_pickle(net, file_path)
    return net
