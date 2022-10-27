import torch
print(torch.cuda.device_count())
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import argparse
import os 
import cv2
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.datasets as datasets
from models import nin,resnet,resnet_more_activation,vgg

import math
import jenkspy

from none import compromised_detection,reset_neurons,test_utils
import get_data
import test_natral_trojan

print(torch.cuda.device_count())

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 and 100 Training')
parser.add_argument('--save_path_root', default='./ckpt/', type=str, metavar='PATH')
parser.add_argument('--model_filepath', default=None, type=str, metavar='PATH')
parser.add_argument('--epoch_num_1', type=int, default=320)
parser.add_argument('--epoch_num_2', type=int, default=20)
parser.add_argument('--round_num', type=int, default=1)
parser.add_argument('--lr_decay_every', type=int, default=80)
parser.add_argument('--arch', type=str, default="nin")
parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--none_lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1)                        
parser.add_argument('--poison_type', type=str, default="single_target")
parser.add_argument('--poison_rate', type=float, default=0.05)
parser.add_argument('--trigger_size', type=int, default=3)
parser.add_argument('--attack_target', type=int, default=0)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--max_reset_fraction', type=float, default=0.05)
parser.add_argument('--lamda_l', type=float, default=0.1)
parser.add_argument('--lamda_h', type=float, default=0.9)
parser.add_argument('--num_for_detect_biased', type=float, default=-1)
parser.add_argument("--m_gpus", action='store_true')

args = parser.parse_args()

if args.model_filepath:
    model_pt_name = args.model_filepath.split("/")[-1]
    model_dir_name = args.model_filepath.split(model_pt_name)[0]
    print("finetune")
else:
    print("train from scratch")
    
    
if args.wd:
    SAVE_PATH = args.save_path_root+args.dataset+"/"+args.arch+"/"+args.poison_type+"_pRate" \
                +str(args.poison_rate)+"_tSize"+str(args.trigger_size)+"_aTarget"+str(args.attack_target)+"_wd" \
                +"/"+"lr"+str(args.none_lr)+"_epochNum1"+str(args.epoch_num_1)+"_epochNum2"+str(args.epoch_num_2)+"_decayEvery"+str(args.lr_decay_every)+"/"
else:
    SAVE_PATH = args.save_path_root+args.dataset+"/"+args.arch+"/"+args.poison_type+"_pRate" \
                +str(args.poison_rate)+"_tSize"+str(args.trigger_size)+"_aTarget"+str(args.attack_target) \
                +"/"+"lr"+str(args.none_lr)+"_epochNum1"+str(args.epoch_num_1)+"_epochNum2"+str(args.epoch_num_2)+"_decayEvery"+str(args.lr_decay_every)+"/"

SAVE_PATH = SAVE_PATH + "_MaxPerLayerF"+str(args.max_reset_fraction)+ "_lamda_l"+str(args.lamda_l)+"_lamda_h"+str(args.lamda_h)+"num_for_detect_biased"+str(args.num_for_detect_biased)+"/"

if not os.path.exists( SAVE_PATH):
    os.makedirs(SAVE_PATH)

logger_path = SAVE_PATH + "log.txt"
logger_file = open(logger_path, 'w')

base_lr = args.base_lr
criterion = nn.CrossEntropyLoss()

num_classes,trainset,trainloader,testloader,testloader_poisoned,trainloader_no_shuffle = get_data.get_dataloaders(args.dataset,args.poison_type,args.trigger_size,args.poison_rate)

if args.arch == "nin":
    print(args.dataset)
    model = nin.Net(num_classes=num_classes,in_channels=3)
    print(model)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.normal_(0, 0.001)
    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
        if key == 'classifier.20.weight':
            params += [{'params':[value], 'lr':0.1 * base_lr, 
                'momentum':0.95, 'weight_decay':0.0001}]
        elif key == 'classifier.20.bias':
            params += [{'params':[value], 'lr':0.1 * base_lr, 
                'momentum':0.95, 'weight_decay':0.0000}]
        elif 'weight' in key:
            params += [{'params':[value], 'lr':1.0 * base_lr,
                'momentum':0.95, 'weight_decay':0.0001}]
        else:
            params += [{'params':[value], 'lr':2.0 * base_lr,
                'momentum':0.95, 'weight_decay':0.0000}]
    optimizer = optim.SGD(params, lr=base_lr, momentum=0.9)
elif args.arch == "resnet18":
    model = resnet.resnet18(num_classes=num_classes)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
elif args.arch == "resnet18_leakyrelu":
    model = resnet_more_activation.resnet18(num_classes=num_classes,activation_function="leakyrelu")
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
elif args.arch == "resnet18_elu":
    model = resnet_more_activation.resnet18(num_classes=num_classes,activation_function="elu")
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
elif args.arch == "resnet18_prelu":
    model = resnet_more_activation.resnet18(num_classes=num_classes,activation_function="prelu")
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
elif args.arch == "resnet18_tanhshrink":
    model = resnet_more_activation.resnet18(num_classes=num_classes,activation_function="tanhshrink")
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
elif args.arch == "resnet18_softplus":
    model = resnet_more_activation.resnet18(num_classes=num_classes,activation_function="softplus")
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
elif args.arch == "vgg11":
    model = vgg.vgg11(num_classes=num_classes)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
elif args.arch == "vgg16":
    model = vgg.vgg16(num_classes=num_classes)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
else:
    print("arch is not supported")

if args.model_filepath:
    try:
        model.load_state_dict(torch.load(args.model_filepath))
    except:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_filepath))
        model = model.module
        
if args.m_gpus:
    model = nn.DataParallel(model)
model.cuda()

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def adjust_learning_rate(optimizer, epoch):
    if epoch%args.lr_decay_every==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

def train_epoch(epoch, loader,freeze_bn=False):
    model.train()
    if freeze_bn:
        model.apply(fix_bn)
    
    for batch_idx, (data, target) in enumerate(loader):

        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(),
                optimizer.param_groups[0]['lr']))
            logger_file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}\n'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item(),
                    optimizer.param_groups[0]['lr']))
    if (epoch+1)%20 == 0:
        torch.save(model.state_dict(), SAVE_PATH.split("latest.")[0] + "epoch" + str(epoch+1) +'.pth')

print("*"*80)
logger_file.write("*"*80+"\n")
test_utils.test(model,testloader,num_classes,criterion,SAVE_PATH,logger_file)
print("-"*80)
logger_file.write("-"*80+"\n")
if args.poison_type != "none":
    test_utils.test_poisoned(model,testloader_poisoned,num_classes,criterion,SAVE_PATH,logger_file,poison_type=args.poison_type,attack_target=args.attack_target)
print("*"*80)
logger_file.write("*"*80+"\n")

if args.dataset not in ["imagenette","trojai"]:
    image_size = 32
else:
    image_size = 224

loader = trainloader
for epoch in range(1, args.epoch_num_1+1):
    adjust_learning_rate(optimizer, epoch)
    train_epoch(epoch,loader)

print("*"*80)
logger_file.write("*"*80+"\n")
test_utils.test(model,testloader,num_classes,criterion,SAVE_PATH,logger_file)
print("-"*80)
logger_file.write("-"*80+"\n")
if args.poison_type != "none":
    test_utils.test_poisoned(model,testloader_poisoned,num_classes,criterion,SAVE_PATH,logger_file,poison_type=args.poison_type,attack_target=args.attack_target)
print("*"*80)
logger_file.write("*"*80+"\n")
    
#test_natral_trojan.test_natral_trojan(SAVE_PATH,image_size,args.arch,logger_file)
    
for param_group in optimizer.param_groups:
    param_group['lr'] = args.none_lr

for round_id in range(1, args.round_num+1):
    
    selected_neuros, poison_sample_index = compromised_detection.analyze_neuros(model,args.arch,args.max_reset_fraction, \
                                                          args.lamda_l,args.lamda_h, num_classes, \
                                                          args.num_for_detect_biased, \
                                                          trainloader_no_shuffle)
    
    indices = list(set([a for a in range(len(trainset))]) - set(poison_sample_index))
    if args.poison_type != "none":
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, indices), batch_size=128, shuffle=True)
    
    model = reset_neurons.reset(model,args.arch,selected_neuros)

    for epoch in range(1, args.epoch_num_2+1):
        adjust_learning_rate(optimizer, epoch)
        train_epoch(epoch,loader,freeze_bn=True)
        print("*"*80)
        logger_file.write("*"*80+"\n")
        test_utils.test(model,testloader,num_classes,criterion,SAVE_PATH,logger_file)
        print("-"*80)
        logger_file.write("-"*80+"\n")
        
        if args.poison_type != "none":
            test_utils.test_poisoned(model,testloader_poisoned,num_classes,criterion,SAVE_PATH,logger_file,poison_type=args.poison_type,attack_target=args.attack_target)
        else:
            test_natral_trojan.test_natral_trojan(SAVE_PATH,image_size,arch,logger_file)
        print("*"*80)
        logger_file.write("*"*80+"\n")
    
    logger_file.flush()
