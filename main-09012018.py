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


def main():


    #########################################################################3
    # INPUTS
    #########################################################################3

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
    dist_backend = 'gloo'
    
    # Url used to setup distributed training
    # v1
    dist_url = 'tcp://127.0.0.1:23456'
    # v2
    #dist_url = 'env://' 
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    #os.environ['MASTER_PORT'] = '23456'
    #os.environ['WORLD_SIZE'] = '4'
    #os.environ['RANK'] = '0'

    # TODO Device to run on (do we need this?????)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    device = ""
    
    #########################################################################3
    # SETUP
    #########################################################################3
   
    ###### Initialize process group

    print("Initialize Process Group...") 
    # v1
    dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=int(sys.argv[1]), world_size=world_size)
    # v2
    #dist.init_process_group(backend=dist_backend, init_method=dist_url)
    # v3 - NOT WORKING - multi-gpus per node with nccl backend
    #dist.init_process_group(backend="nccl", init_method="file:///home/ubuntu/distributed_tutorial/trainfile", rank=int(sys.argv[1]), world_size=2)

    ###### Initialize Model

    print("Initialize Model...")
    model = models.resnet18(pretrained=False).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), starting_lr, momentum=0.9, weight_decay=1e-4)

    ###### Initialize Dataloaders

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


    #########################################################################3
    # TRAINING LOOP
    #########################################################################3

    best_prec1 = 0

    for epoch in range(num_epochs):
        
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(starting_lr, optimizer, epoch)

        # train for one epoch
        print("\nBegin Training Epoch {}".format(epoch+1))
        train(train_loader, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        print("Begin Validation @ Epoch {}".format(epoch+1))
        prec1 = validate(val_loader, model, criterion, device)

        # remember best prec@1 and save checkpoint if desired
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print("Epoch Summary: ")
        print("\tEpoch Accuracy: {}".format(prec1))
        print("\tBest Accuracy: {}".format(best_prec1))



def train(train_loader, model, criterion, optimizer, epoch, device):
    
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

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        #dist.all_reduce_multigpu(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def validate(val_loader, model, criterion, device):

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
