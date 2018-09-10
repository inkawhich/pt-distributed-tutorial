import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


#########################################################################3
# MAIN
#########################################################################3
def main():

    """
    #########################################################################3
    ## INPUTS

    Here is where we will define the inputs for the run. Some of the inputs are standard
    model training inputs such as batch size and number of training epochs, and some are
    specific to our distributed training task.i

    - batch_size - batch size for *each* process in the distributed training group. Total batch size 
                   across distributed model is batch_size*world_size ????????

    - workers - number of worker threads used with the dataloaders

    - num_epochs - number of epochs to train for

    - starting_lr - starting learning rate for training

    - world_size - number of processes in the distributed training environment

    - dist_backend - backend to use for distributed training communication (i.e. NCCL, Gloo, MPI, etc.)

    - dist_url - url to specify the initialization method of the process group. This may 
                 contain the IP address and port of the rank0 process or be a non-existant file on a shared file system.

    """

    print("Collect Inputs...")

    # Batch Size for training and testing
    batch_size = 64

    # Number of worker threads for dataloading
    workers = 4

    # Number of epochs to train for
    num_epochs = 2

    # Starting Learning Rate
    starting_lr = 0.1

    # Number of distributed processes
    world_size = 2 
    
    # Distributed backend type
    #dist_backend = 'nccl'
    dist_backend = 'gloo'
    
    # Url used to setup distributed training
    # v1
    #dist_url = "tcp://18.205.21.252:23456"
    dist_url = "tcp://172.31.22.234:23456"
    #dist_url = "tcp://172.17.0.1:23456"
    # v2
    #dist_url = "file:///home/ubuntu/distributed_tutorial/trainfile"

   
    """
    #########################################################################3
    ### Initialize process group

    One of the most important parts of distributed training in PyTorch is to properly setup
    the process group, which is the **first** step in initializing the torch.distributed package.
    To do this, we will use the `torch.distributed.init_process_group` function which takes
    several inputs. First, a *backend* input which specifies the backend to use (i.e. NCCL, Gloo, MPI, etc.).
    An *init_method* input which is either a url containing the address and port
    of the rank0 machine or a path to a non-existant file on the shared file system. Note, 
    to use the file init_method, all machines must have access to the file, similarly for the
    url method, all machines must be able to communicate on the network so make sure to configure
    any firewalls and network settings to accomodate. The init_process_group function also takes
    *rank* and *world_size* arguments which specify the rank of this process when run and the number
    of processes in the collective, respectively. It is important to note that this is a blocking
    function, meaning program execution will wait at this function until *world_size* processes have
    joined the process group.

    Another important step, especially when each node has multiple gpus is to set the *local_rank* of
    this process. For example, if you have two nodes, each with 8 GPUs and you wish to train with all
    of them then $world_size=16* and each node will have a process with local rank 0-7. This local_rank
    is used to set the device (i.e. which GPU to use) for the process and later used to set the device when 
    creating a distributed data parallel model. It is also recommended to use NCCL backend in this 
    hypothetical environment as NCCL is preferred for multi-gpu nodes. 
    
    """
    print("Initialize Process Group...") 
    # v1 - init with url
    dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=int(sys.argv[1]), world_size=world_size)
    # v2 - init with file
    #dist.init_process_group(backend="nccl", init_method="file:///home/ubuntu/pt-distributed-tutorial/trainfile", rank=int(sys.argv[1]), world_size=world_size)
    
    local_rank = int(sys.argv[2])
    dp_device_ids = [local_rank]
    torch.cuda.set_device(local_rank)


    """
    #########################################################################3
    ### Initialize Model
    
    (?) - Does DPP also handle averaging of gradients?
    
    The next major step is to initialize the model to be trained. Here, we will use
    a resnet18 model from `torchvision.models` but any model may be used.
    First, we initialize the model and place it in GPU memory.
    The next step, which is very important for our distributed training example, is to 
    make the model `DistributedDataParallel`, which handles the distribution of the data
    to and from the model. Also notice we pass our device ids list as a parameter which contains
    the local rank (i.e. GPU) we are using. Finally, we specify the loss function and 
    optimizer to train with.

    """

    print("Initialize Model...")
    model = models.resnet18(pretrained=False).cuda()
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), starting_lr, momentum=0.9, weight_decay=1e-4)

    """
    #########################################################################3
    ### Initialize Dataloaders

    (?) - mention about pin_memory arg in the dataloaders

    The last step in preparation for the training is to specify which dataset to use. Here
    we use the STL10 dataset from torchvision.datasets.STL10. The STL10 dataset is a 
    10 class dataset of 96x96px images. For use with our model, notice we resize the images
    to 224x224px in the transform. One distributed training specific item in this section
    is the use of the `DistributedSampler` for the training set, which is designed to be
    used in conjunction with `DistributedDataParallel` models. This object handles the 
    partitioning of the dataset across the distributed environment so that not all models are training
    on the same subset of data, which would be counterintuitive. Finally, we create the `DataLoader`'s 
    which are responsible for feeding the data to the processes. 

    """

    print("Initialize Dataloaders...")
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
    valset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    #train_sampler = None

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=workers, pin_memory=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)


    """
    #########################################################################
    ### Training Loop
    
    Finally, the last step is to execute the training loop. We have already done most 
    of the work for setting up the distributed training so this training loop has
    almost no artifacts of distributed training. The one detail specific to our task
    is the setting the current epoch count in the `DistributedSampler`, as the sampler
    shuffles the data going to each process deterministically based on epoch. After
    updating the sampler, the loop runs a full training epoch, runs a full validation
    step then prints the performance of the current model against the best performing
    so far. After training for num_epochs, the loop exits and the tutorial is complete. 
    Notice, since this is an exercise we are not saving models but one may wish to 
    keep track of the best performing model then save it at the end of training.

    """

    best_prec1 = 0

    for epoch in range(num_epochs):
        
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(starting_lr, optimizer, epoch)

        # train for one epoch
        print("\nBegin Training Epoch {}".format(epoch+1))
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        print("Begin Validation @ Epoch {}".format(epoch+1))
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint if desired
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print("Epoch Summary: ")
        print("\tEpoch Accuracy: {}".format(prec1))
        print("\tBest Accuracy: {}".format(best_prec1))



#########################################################################3
# Train Fxn
#########################################################################3
def train(train_loader, model, criterion, optimizer, epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Average gradients across all workers 
        average_gradients(model)

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

#########################################################################3
# Average Gradients Fxn
#########################################################################3
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        # TODO dist.all_reduce_multigpu(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


#########################################################################3
# Validate Fxn
#########################################################################3
def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


#########################################################################3
# Helper Fxn/Classes
#########################################################################3
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
