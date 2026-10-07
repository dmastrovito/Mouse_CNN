#!/usr/bin/env python
# coding: utf-8

# In[1]:

#exec(open("construct_mousenet_architecture.py").read())

import sys
sys.path.append('cmouse/')
sys.path.append('mouse_cnn/')
from anatomy import *
from architecture import *
import pandas as pd


#get_ipython().magic(u'matplotlib inline')

import os
ccf_version = "ccf_2017"
from config import *

#sys.path.append('/allen/programs/mindscope/workgroups/tiny-blue-dot/mouse_connectivity_models_2020')
# In[2]:

recurrent = True
architecture = Architecture(ccf_version = ccf_version,recurrent = recurrent)
anet = gen_anatomy(architecture,recurrent = recurrent)


# In[3]:


#anet.draw_graph()


# In[4]:



from network import Network
net = Network()
net_params = net.construct_from_anatomy(anet, architecture,recurrent = recurrent)
df = pd.DataFrame(net_params)


from network import save_network_to_pickle
if not recurrent:
    file = 'mousenet_inputsize'+ str(INPUT_SIZE[1])+"_"+ccf_version
    save_network_to_pickle(net, os.path.join("models",file + '.pkl'))
else:
    file = 'recurrent_mousenet_inputsize'+str(INPUT_SIZE[1])+"_"+ccf_version
    save_network_to_pickle(net,os.path.join("models",file +'.pkl'))
df.to_csv(file+".csv")

