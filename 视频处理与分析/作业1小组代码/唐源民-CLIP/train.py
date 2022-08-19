
   
# -*- coding: utf-8 -*-  

"""
Created on 2021/07/14
@author: Ruoyu Chen
"""

import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from dataset import Dataset

from prettytable import PrettyTable
from models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, VGG16, ViT_CLIP

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def define_Loss_function(loss_name, pos_loss_weight=None, weight = None):
    if loss_name == "BCELoss":
        Loss_function = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_loss_weight)
    return Loss_function

def define_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.01)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.01)
    return optimizer

def TPFN(model, validation_loader, attribute_num, attribute_name, thresh, device):
    """
    Compute the TP, FN, FP, TN
    Precision = TP / (TP + FP)
    Recell = TP / (TP + FN)
    """
    model.eval()

    corrects = np.zeros((attribute_num, 4))   # [Batch, (TN,FN,FP,TP)]
    with torch.no_grad():
        for i, (data,labels) in enumerate(validation_loader):
            data = data.to(device)
            labels = labels.to(device)

            outputs = sigmoid(model(data))

            ii = 0
            for output, label in zip(outputs.t(), labels.t()):
                # output: Torch_size(batch)
                # label: Torch_size(batch)
                output_label = (output>thresh).int()
                # if ii ==1:
                #     print(output)

                results = output_label * 2 + label
                TN_n = len(torch.where(results==0)[0])
                FN_n = len(torch.where(results==1)[0])
                FP_n = len(torch.where(results==2)[0])
                TP_n = len(torch.where(results==3)[0])

                assert len(results) == TN_n + FN_n + FP_n + TP_n

                corrects[ii][0] += TN_n; corrects[ii][1] += FN_n
                corrects[ii][2] += FP_n; corrects[ii][3] += TP_n
                ii += 1
    
    table = PrettyTable(["Attribute Name", "TP", "FN", "FP", "TN","ACC"])
    
    for i in range(attribute_num):
        table.add_row([str(i)+". "+attribute_name[i],
                       corrects[i][3],  # TP
                       corrects[i][1],  # FN
                       corrects[i][2],  # FP
                       corrects[i][0],  # TN
                       "%.4f"%((corrects[i][3]+corrects[i][0])/(corrects[i][0]+corrects[i][1]+corrects[i][2]+corrects[i][3]))    # ACC
                       ]) 
    print(table)

def main(args):
    learning_rate = 0.01


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    device = torch.device("cuda")

    # Configuration file

    # model save path
    model_save_path = os.path.join("checkpoint", "ViT_CLIP")
    mkdir(model_save_path)
    model = ViT_CLIP()

    # Dataloader
    train_dataset = Dataset(dataset_list=args.dataset_list)
    train_loader = DataLoader(train_dataset,batch_size=10, shuffle=True)

    # GPU
    if torch.cuda.is_available():
        model = model.cuda()
    # Multi GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Loss
    # weight = torch.Tensor([cfg.ATTR_LOSS_WEIGHT for i in range(cfg.BATCH_SIZE)]).to(device)
    # weight = torch.Tensor(cfg.ATTR_LOSS_WEIGHT).to(device)
    Loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([3.1, 1.795454545, 4.347826087, 8.461538462, 10.18181818, 14.375, 6.6875, 0.481927711]).cuda())
    # Loss_function = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.01)

    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    for i in range(1,101):
        scheduler.step()

        model.train()
        
        for ii, (data,label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            
            loss = Loss_function(output, label)
            # print(output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i* len(train_loader) + ii

            if iters % 1 == 0:
                print(
                    "\033[32m{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + 
                    "\033[0mtrain epoch " + 
                    "\033[34m{} ".format(i) +
                    "\033[0miter " + 
                    "\033[34m{} ".format(ii) + 
                    "\033[0mloss " + 
                    "\033[34m{}.".format(loss.item())
                )

        if i % 10 == 0:
            TPFN(model, train_loader, 8, ["混乱", "空洞", "分裂", "受伤", "流动", "趋中", "整合", "能量"], 0.5, device)
            torch.save(model.state_dict(), os.path.join(model_save_path,"backbone-item-epoch-"+str(i)+'.pth'))


def parse_args():
    parser = argparse.ArgumentParser(description='HW1')
    parser.add_argument('--dataset-root', type=str, default='./data',
                        help='GPU device')
    parser.add_argument('--dataset-list', type=str, default='./train.txt',
                        help='GPU device')
    parser.add_argument('--gpu-device', type=str, default="3",
                        help='GPU device')
    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    sigmoid = nn.Sigmoid()
    sigmoid.to(torch.device("cuda"))
    args = parse_args()
    main(args)