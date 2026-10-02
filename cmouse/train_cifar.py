import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
from cifar_config import *
from train_config import *
import network
from mousenet_complete_pool import MouseNetCompletePool, debug_memory
#from fsimilarity import *
import random
#import wandb
import argparse
from numpy.random import default_rng

RUN = 1
suffix = "_run_"+str(RUN)
SEED = 42 + RUN
loss = nn.CrossEntropyLoss()

rng = default_rng(SEED)
 

def plot_states(states,outdir,suffix=""):
    
    #plt.clf()
    for r,region in enumerate(list(states.keys())):
        plt.clf()
        #plt.subplot(4,6,r+1)
        plt.plot(np.arange(1,len(states[region])+1),np.array(states[region]),marker='.')
        plt.title(region)
        plt.savefig(os.path.join(outdir,"_".join((region.replace("/","_"),'states_over_time_new_initialization_Adam_bias_fullsteps_tanh'+suffix+'.png'))))
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
        plt.savefig(os.path.join(outdir,"Validation_loss_new_initialization_Adam_bias_fullsteps_tanh"+suffix+".png"))
    else:
        plt.plot(loss)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "Training_loss_new_initialization_Adam_bias_fulsteps_tanh"+suffix+".png"))

    

def train(args, model, device, train_loader, optimizer, epoch,training_loss = None,recurrent = False,nsteps = None,step_range=None,suffix=""):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()
    
       
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        model.reset(BATCH_SIZE, device)
        data, labels = data.to(device), labels.to(device)
        if recurrent:
            n_steps = rng.integers(low=step_range[0], high=step_range[1], size=1)[0]   
        else:
            n_steps = 6
        out,states = model(data,n_steps = n_steps,track_states = True)
        l = loss(out, labels)
        l.backward()
        
        optimizer.step()
        training_loss.append(l.detach().cpu().numpy())
        if batch_idx %10 ==0:
            debug_memory()
            plot_states(states,RESULT_DIR,suffix=suffix) ##########
            plot_loss(training_loss,RESULT_DIR,suffix=suffix) ############
            print(n_steps,epoch,l.detach().cpu().numpy(),batch_idx, "Of",len(train_loader))
        
        if batch_idx%100 ==0:
            torch.save(model.state_dict(),os.path.join(RESULT_DIR,RESULT_DIR+"_"+suffix+'.pth')) ####################
    return training_loss

    
   

def test(args, model, device, test_loader, epoch,best_acc =0, training_loss = None, validation_loss = None,recurrent = False,nsteps = None,step_range=None,suffix=""):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      correct = 0
      for data, target in test_loader:
          # Load the input features and labels from the test dataset
          data, target = data.to(device), target.to(device)
          model.reset(BATCH_SIZE,device)
          # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
          if recurrent:
              n_steps = rng.integers(low=step_range[0], high=step_range[1], size=1)[0]   
          else:
              n_steps = 6
          output,states = model(data, n_steps=n_steps)                 
          
          # Compute the loss sum up batch loss
          #test_loss += F.nll_loss(output, target, reduction='sum').item()
          test_loss = loss(output, target)
          pred = output.max(1, keepdim=True)[1]
          correct += pred.eq(target.view_as(pred)).sum().item()         
          validation_loss.append(test_loss.cpu().numpy())
      acc = 100. * correct / len(test_loader.dataset)
      plot_loss(validation_loss,RESULT_DIR, accuracy= acc, validation = True,acc = " ".join(("epoch",str(epoch), "acc",str(acc))),suffix=suffix)
      
      print(acc)
    

    # Save checkpoint.
    
    print(acc, "best_acc",best_acc)
    if epoch == 0 or acc > best_acc:
        save_dir = RESULT_DIR 
        print('Saving to '+save_dir+"...")
        state = {
            'state_dict': model.state_dict(),
            'best_acc': acc,
            'epoch': epoch,
            'training_loss':training_loss,
            'validation_loss':validation_loss,
        }
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if epoch == 0:
            torch.save(state, save_dir + '%s_init_run_%s.pt'%(SEED,RUN))
        else:
            torch.save(state, save_dir + '%s_%s_run_%s.pt'%(SEED, acc,RUN))
        best_acc = acc

    return best_acc


def get_data_loaders(VGG = False,Grayscale = False,batch_size=None):
    global  BATCH_SIZE
    if batch_size is not None:
        BATCH_SIZE = batch_size
    
    # preparing input transformation
    if INPUT_SIZE[0]==3:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize(INPUT_SIZE[1:],interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470,0.2435,0.2616)),])
                                                     #old std was wrong (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([
                            transforms.Resize(INPUT_SIZE[1:],interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470,0.2435,0.2616)),])
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
        if Grayscale:
            transform_train.transforms.append(transforms.Grayscale(3))
            transform_test.transforms.append(transforms.Grayscale(3))
            
        vgg_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize( 
               (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
            )
        ])
    elif INPUT_SIZE[0]==2:
        class GBChannels():
            def __call__(self, tensor):
                return tensor[1:3,:,:]
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize(INPUT_SIZE[1:]),
                                transforms.ToTensor(),
                                GBChannels(),
                                transforms.Normalize((0.4914, 0.4822), (0.2023, 0.1994))])
        transform_test = transforms.Compose([transforms.Resize(INPUT_SIZE[1:]),
                                transforms.ToTensor(),
                                GBChannels(),
                                transforms.Normalize((0.4914, 0.4822), (0.2023, 0.1994))])
    else:
        raise Exception('Number of input channel should be 2 or 3!')
    
    # load dataset
    if DATASET == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                        download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                        download=True, transform=transform_test)
        if VGG:
            vgg_test = torchvision.datasets.CIFAR10(root = DATA_DIR, train = False, 
                                        download = True, transform = vgg_transform)
    
    elif DATASET == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True,
                                        download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False,
                                        download=True, transform=transform_test)
    else:
        raise Exception('DATASET should be cifar10 or cifar100')
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=0,drop_last = True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=0,drop_last = True)
    if VGG:
        vgg_test_loader = torch.utils.data.DataLoader(vgg_test, batch_size=BATCH_SIZE,
                                            shuffle = False, num_workers = 0, drop_last = True)
    # WandB - wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    #if not WANDB_DRY:
    #    wandb.watch(mousenet, log="all") 
    if VGG:
        return train_loader, vgg_test, vgg_test_loader
    else:
        return train_loader, test_loader

def adjust_learning_rate(config, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every LR_EPOCHS"""
    lr = LR * (0.1 ** (epoch // LR_EPOCHS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if USE_WANDB:
        config.update({'lr': lr}, allow_val_change=True)


def set_save_dir(directory):
    global RESULT_DIR
    
    RESULT_DIR = directory
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

def main():
    
    recurrent = False
    #nsteps = 'baseline'
    #step_range = None
    nsteps = 'sampled'
    step_range = (30,40)
    set_save_dir(recurrent,nsteps = nsteps)
    

    training_loss = []
    validation_loss = []
    best_acc = 0
    device = torch.device("cuda")
    train_loader, test_loader = get_data_loaders()
    #device = torch.device('cpu')
    # print the shape of the input
    print(RESULT_DIR)
    
    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(SEED)       # python random seed
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED)    # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    # get the mouse network
    
    
    #net_name = 'network_(%s,%s,%s)'%(INPUT_SIZE[0],INPUT_SIZE[1],INPUT_SIZE[2])
    #architecture = Architecture(data_folder=DATA_DIR)
    #net = gen_network(net_name, architecture)
    #mousenet = MouseNetCompletePool(net, mask=MASK)
    
   
    
    
    if recurrent:
        net = network.load_network_from_pickle('../network_complete_updated_number(3,64,64)_edited_sigma_recurrent.pkl')
    else:
        net = network.load_network_from_pickle('../network_complete_updated_number(3,64,64).pkl')
    mousenet = MouseNetCompletePool(net, recurrent = recurrent)
    
    mousenet.to(device)    
    optimizer = optim.SGD(mousenet.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=5e-4)
    
    
    #optimizer = optim.Adam(mousenet.parameters())
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2*EPOCHS//3, 9*EPOCHS//10], gamma=0.2)

    config = None
    '''
    best_acc = test(config, mousenet, device, test_loader, 0,best_acc = best_acc, training_loss = training_loss, \
                   validation_loss = validation_loss,recurrent = recurrent,nsteps = nsteps,step_range = step_range)  
    '''    
    #debug_memory()
    for epoch in range(1, EPOCHS + 1):  # loop over the dataset multiple times
        #adjust_learning_rate(config, optimizer, epoch)
        print(epoch)  
        training_loss = train(config, mousenet, device, train_loader, optimizer, epoch, training_loss = training_loss,\
                                recurrent = recurrent,nsteps = nsteps,step_range = step_range,suffix=suffix)
        debug_memory()
        best_acc = test(config, mousenet, device, test_loader, epoch,best_acc = best_acc, training_loss = training_loss, \
                        validation_loss = validation_loss,recurrent = recurrent,nsteps = nsteps,step_range = step_range,suffix=suffix)  
        #debug_memory()
        #break
        scheduler.step()
    if recurrent:
        if nsteps == 'sampled':
            outfile = "_".join()
            torch.save(mousenet.state_dict(),"mousenet_cifar_trained_recurrent"+suffix+".sav")
        else:
            torch.save(mousenet.state_dict(),"mousenet_cifar_trained"+suffix+".sav")
    
    print('Finished Training')
    return


if __name__ == "__main__":
    main()
    
    




