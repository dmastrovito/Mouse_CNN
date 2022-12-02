import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
from cifar_config import *
from train_config import *
from anatomy import *
import network
from mousenet_complete_pool import MouseNetCompletePool, debug_memory
#from fsimilarity import *
import random
#import wandb
import argparse
from numpy.random import default_rng

parser = argparse.ArgumentParser(description='PyTorch %s Training' % DATASET)
parser.add_argument('--seed', default = 42, type=int, help='random seed')
parser.add_argument('--mask', default = 3, type=int, help='if use Gaussian mask')
args = parser.parse_args()
SEED = args.seed
MASK = args.mask
RUN_NAME_MASK = 'ReLU_eachstep_multiplicative_recurrence_%s_%s'%(MASK, RUN_NAME)
#RUN_NAME_MASK = 'sampled_inference_time_multiplicative_recurrence_%s_%s'%(MASK, RUN_NAME)
RUN_NAME_MASK_SEED = '%s_seed_%s'%(RUN_NAME_MASK, SEED)



rng = default_rng(SEED)
    

    

def train(args, model, device, train_loader, optimizer, epoch,training_loss = None,recurrent = False,nsteps = None,step_range=None):
    # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
    for batch_idx, (data, target) in enumerate(train_loader):
        # Load the input features and labels from the training dataset
        data, target = data.to(device), target.to(device)
        # Reset the gradients to 0 for all learnable weight parameters
        #debug_memory()
        optimizer.zero_grad()
        # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
        if recurrent and nsteps == 'sampled':
            n_steps = rng.integers(low=step_range[0], high=step_range[1], size=1)[0]    
            #print(n_steps)
        else:
            n_steps = None
        output,_ = model(data,n_steps = n_steps)
        # Define our loss function, and compute the loss
        loss = F.cross_entropy(output, target)
        #print(loss)
        # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
        loss.backward()
        
        '''
        for parameter in list(model.named_parameters()):
            if  "BNs" not in parameter[0] and 'mask' not in parameter[0] and parameter[1].grad == None:
                print(parameter[0],"zero grad")
            elif ("LGNd" in parameter[0] or 'VISpor5' in parameter[0]) and "BNs" not in parameter[0] and 'mask' not in parameter[0]:
                print (batch_idx,parameter[0],"data",torch.mean(parameter[1].data))
                print (batch_idx,parameter[0],"grad",torch.mean(parameter[1].grad))
        '''
        
        optimizer.step()
        
        '''
        named_params = list(model.named_parameters())
        zerod = []
        with torch.no_grad():
            for p,param in enumerate(named_params):
                if any(x in param[0] for x in ('weight','bias')):
                    layername = [name for name in model.areas if name in param[0]]
                    if len(layername) > 1:
                        idx = [param[0].find(name) for name in layername]
                        order = sorted(range(len(idx)), key=lambda k: idx[k])
                        source_name =  layername[order[0]]
                        target_name = layername[order[1]]
                        if model.areas.index(source_name) > model.areas.index(target_name):
                            param[1].data = torch.zeros(param[1].shape,device=device)
        '''
        # Update the neural network weights
        
        training_loss.append(loss.clone().detach().cpu().numpy())
     
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        running_loss += loss.item()
        if batch_idx % LOG_INTERVAL == LOG_INTERVAL-1:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, batch_idx + 1, running_loss / LOG_INTERVAL))
            if USE_WANDB:
                wandb.log({
                "running loss": running_loss/LOG_INTERVAL,
                "running acc": 100.*correct/total})
            running_loss = 0.0
        del data
        del target   
        torch.cuda.empty_cache()  
    return training_loss


def test(args, model, device, test_loader, epoch,best_acc =0, training_loss = None, validation_loss = None,recurrent = False,nsteps = None,step_range=None):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    
    #example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            # Load the input features and labels from the test dataset
            data, target = data.to(device), target.to(device)
            # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
            if recurrent and nsteps == 'sampled':
                n_steps = rng.integers(low=step_range[0], high=step_range[1], size=1)[0]    
            else:
                n_steps = None
            output,_ = model(data,n_steps = n_steps)
                    
            # Compute the loss sum up batch loss
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            test_loss = F.cross_entropy(output, target)
            validation_loss.append(test_loss)
            # Get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # WandB - Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            #example_images.append(wandb.Image(
            #    data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))
            del data
            del target
            torch.cuda.empty_cache()

    # Save checkpoint.
    
    acc = 100. * correct / len(test_loader.dataset)
    print(acc, "best_acc",best_acc)
    if epoch == 0 or acc > best_acc:
        save_dir = RESULT_DIR + '/' + RUN_NAME_MASK +'/'
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
            torch.save(state, save_dir + '%s_init.pt'%(SEED))
        else:
            torch.save(state, save_dir + '%s_%s.pt'%(SEED, acc))
        best_acc = acc

    return best_acc


def get_data_loaders():
    # preparing input transformation
    if INPUT_SIZE[0]==3:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize(INPUT_SIZE[1:]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([
                            transforms.Resize(INPUT_SIZE[1:]),transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
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
     
    # WandB - wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    #if not WANDB_DRY:
    #    wandb.watch(mousenet, log="all") 
    return train_loader, test_loader

def adjust_learning_rate(config, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every LR_EPOCHS"""
    lr = LR * (0.1 ** (epoch // LR_EPOCHS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if USE_WANDB:
        config.update({'lr': lr}, allow_val_change=True)


def set_save_dir(recurrent = False,nsteps = None):
    global RESULT_DIR
    
    if recurrent:
        RESULT_DIR = os.path.join(RESULT_DIR,'recurrent',nsteps)
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

def main():
    
    recurrent = True
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
                                recurrent = recurrent,nsteps = nsteps,step_range = step_range)
        debug_memory()
        best_acc = test(config, mousenet, device, test_loader, epoch,best_acc = best_acc, training_loss = training_loss, \
                        validation_loss = validation_loss,recurrent = recurrent,nsteps = nsteps,step_range = step_range)  
        #debug_memory()
        #break
        scheduler.step()
    if recurrent:
        if nsteps == 'sampled':
            outfile = "_".join()
            torch.save(mousenet.state_dict(),"mousenet_cifar_trained_recurrent.sav")
        else:
            torch.save(mousenet.state_dict(),"mousenet_cifar_trained.sav")
    
    print('Finished Training')
    return


if __name__ == "__main__":
    main()
    
    




