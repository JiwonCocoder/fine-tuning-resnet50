#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function
from pl_bolts.models.self_supervised import SimCLR
import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models

# import models
from utils import progress_bar

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet50", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=50, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')

parser.add_argument('--dataset', default="MLCC", type=str,
                    help='')

parser.add_argument('--pretrain', default="supervised-imagenet", type=str,
                    help='')

parser.add_argument('--mixup', default=True, type=boolean_string,
                    help='')

parser.add_argument('--limit_data', default=False, type=boolean_string,
                    help='')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
0
if args.seed != 0:
    torch.manual_seed(args.seed)

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['MLCC'] = [0.1778, 0.04714, 0.16583]
std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['MLCC'] = [0.26870, 0.1002249, 0.273526]
# Data
print('==> Preparing data..')

def get_transform_cifar(train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean['cifar10'], std['cifar10'])])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean['cifar10'], std['cifar10'])])
def get_transform_MLCC(train=True):
    if train:
        return transforms.Compose([
                                   transforms.RandomVerticalFlip(p=0.5),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean['MLCC'], std['MLCC'])])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean['MLCC'], std['MLCC'])])
"""
if args.augment:
    transform_train = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    print("auged")
else:
    transform_train = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        
    ])
    print("no auged")
"""

transform_test = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

if args.dataset == "CIFAR10" :
    print("CIFAR10 Dataset\n")
    trainset = datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform_train)

    testset = datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
    
elif args.dataset == "MLCC" :
    print("MLCC Dataset\n")
    trainset = torchvision.datasets.ImageFolder(root="./sup_set/Train", transform=get_transform_MLCC(train=True))
    testset = torchvision.datasets.ImageFolder(root="./sup_set/Test", transform=get_transform_MLCC(train=False))


##################데이터 4000개만 쓰기###########################
if args.limit_data == True:
    trainset.data = trainset.data[:4000]
    trainset.targets = trainset.targets[:4000]
    print("only 4000 data")
#############################################################
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=8) 
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    checkpoint = torch.load('./checkpoint/' + args.name + 'pt')                        
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    if args.model == "ResNet50" :
        if args.pretrain == "supervised-imagenet":
            print("Supervised Imagenet Pretrained Resenet50")
            net = torchvision.models.resnet50(pretrained=True)
            tmp = net.fc.in_features
            net.fc = nn.Linear(tmp,10)
        elif args.pretrain == "simCLR-imagenet" :
            # load resnet50 pretrained using SimCLR on imagenet
            print("SimCLR Imagenet Pretrained Resenet50")
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
            net = simclr.encoder
            net = torchvision.models.resnet50(pretrained=True)
            tmp = net.fc.in_features
            net.fc = nn.Linear(tmp,10)
        elif args.pretrain == "scratch" :
            print("scratch Resenet50")
            net = torchvision.models.resnet50(pretrained=False)
            tmp = net.fc.in_features
            net.fc = nn.Linear(tmp,10)
            
    else :
        net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
#                       weight_decay=args.decay)
#samsung setting below
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay, nesterov=True)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
    

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        if args.mixup :
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                        args.alpha, use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                        targets_a, targets_b))
                                        
        
        outputs = net(inputs)# kr
        if args.mixup :
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam) #kr
        else :
            loss = criterion(outputs, targets) #kr

        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        #correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
        #            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        correct += predicted.eq(targets.data).cpu().sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            inputs = Variable(inputs)
        target = Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    ########################################################################
    # """decrease the learning rate at 100 and 150 epoch"""
    # lr = args.lr
    # if epoch >= 100:
    #     lr /= 10
    # if epoch >= 150:
    #     lr /= 10
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    ########################################################################
    #samsung setting below
    lr = args.lr
    if epoch % 50 == 0:
        lr /= 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])
