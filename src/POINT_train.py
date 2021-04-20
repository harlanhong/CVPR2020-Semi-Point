from __future__ import print_function

import argparse
# from vps import VPS
import pdb
from collections import defaultdict
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision import transforms as T
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
seed = random.randint(1,10000)
print('=============seed ;{}==============='.format(seed))
setup_seed(seed)
from dataset import *
from preprocessor import *
from relationNetwork import *
import torch.autograd
from sampler import *
import math
import cv2
from tqdm import tqdm  
import copy
import time
from GetMapAndCMC import *
from option import args
from torch.optim import *
from util.DataProvider import DataProvider

args.cuda = not args.no_cuda and torch.cuda.is_available()

VIPDataset = SemiNetDataset(root=args.dataset_path,index=args.index_name)

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
train_transformer1 = T.Compose([
        T.RandomResizedCrop((224),(0.8, 0.85)),
        #T.CenterCrop(224),

        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer
        ])

train_transformer2 = T.Compose([
        # T.ToPILImage(),
        T.RandomResizedCrop((224), (0.8, 0.85)),
        #T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer
        ])

train_transformer3 = T.Compose([
        # T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        normalizer
        ])

val_transformer1 = T.Compose([T.Resize((224,224)),
                                T.ToTensor(),
                                normalizer
                                ])

val_transformer2 = T.Compose([T.Resize((224,224)),
                                T.ToTensor(),
                                normalizer
                                ])

val_transformer3 = T.Compose([T.Resize((224, 224)),
                                T.ToTensor(),
                                normalizer
                                ])
#load train_loader and val_loader
lable_train_set = VIPDataset.label_train  
val_set = VIPDataset.val   
test_set = VIPDataset.test    
train_num = VIPDataset.num_label_train 
val_num = VIPDataset.num_val
test_num = VIPDataset.num_test
train_loader = DataLoader(
    RelationNetCPreprocessor(lable_train_set, isTrain=True, 
                     transform1=train_transformer1,transform2=train_transformer2,transform3=train_transformer3),
                     sampler = RelationNetRandomFaceSampler(data_source=lable_train_set,
                                                     num_instances=args.ni),
                     batch_size=args.batch_size,num_workers=args.nw,
                     pin_memory=True)

val_loader = DataLoader(
    RelationNetCPreprocessor(val_set, isTrain=False,
                     transform1=val_transformer1, transform2=val_transformer1,transform3=val_transformer3),
                     sampler = RelationNetTestSampler(data_source=val_set),
                     batch_size=args.batch_size, num_workers=args.nw,
                     pin_memory=True)

test_loader = DataLoader(
    RelationNetCPreprocessor(test_set, isTrain=False,
                     transform1=val_transformer1, transform2=val_transformer1,transform3=val_transformer3),
                     sampler = RelationNetTestSampler(data_source=test_set),
                     batch_size=args.batch_size, num_workers=args.nw,
                     pin_memory=True)

#The path and name saved by the model
saveDir='../models/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
model_name=args.save_name+"_"+str(seed)+'.pkl'
labelProvider = DataProvider(train_loader)

print(model_name)
#load model
model = POINT_Module_1b(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
model_fc = POINT_Module_1b_fc(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
best=0
if args.cuda:
    model = model.cuda()
    model_fc = model_fc.cuda()
alpha = 0.5
lossF = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer,step_size = args.lr_decay,gamma = 0.5)

# weights = [1/7,1]
# class_weights = torch.FloatTensor(weights).cuda()
#Decay the learning rate
def adjust_learning_rate(opt, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr * (0.5 ** (epoch // args.lr_decay))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


#Traing
def train(epoch, val_mAP,bestResult,test_mAP):
    accuracy=0
    num=0
    total_loss=0
    for batch_idx in tqdm(range(args.training_step)):
        # break
        src,face, faceCont,coor,target, ImgId = labelProvider.next()
        model.train()
        if args.cuda:
            src,face,faceCont,coor, target = src.cuda(),face.cuda(),faceCont.cuda(),coor.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(src,face,faceCont,coor)
        celoss = lossF(output,target)
        celoss.backward()
        optimizer.step()
        total_loss+=celoss.item()
        prediction = torch.max(F.softmax(output), 1)[1]
        pred_y = prediction.data.cpu().numpy().squeeze()
        target_y = target.data.cpu().numpy()
        accuracy = sum(pred_y == target_y)+accuracy
        num=target.size(0) +num
       
    print('Epoch: {} Time: {} CELoss: {:.6f} Val mAP: {:.4f} Test mAP {:.4f} Train Acc: {:.4f}'.format(
                epoch,time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),total_loss/len(train_loader), val_mAP,test_mAP,train_acc))
    if epoch%args.interval==0 and epoch!=0:
        with torch.no_grad():
        #Assign test model parameters
            model.eval()
            model_fc.eval()
            dict_train=model.state_dict()
            dict_fc = model_fc.state_dict().copy()
            pretrained_dict = {k: v for k, v in dict_train.items() if k in dict_fc}#filter out unnecessary keys 
            dict_fc.update(pretrained_dict)
            model_fc.load_state_dict(dict_fc)

            # extract the feature first
            gfeat_dic = defaultdict(list)
            pfeat_dic= defaultdict(list)
            label_dic = defaultdict(list)
            for val_batch_idx, (val_src,val_face, val_faceCont,val_coor, val_label, val_ImgId) in enumerate(tqdm(val_loader)):
                if args.cuda:
                    val_src,val_face,val_faceCont, val_coor= val_src.cuda(),val_face.cuda(),val_faceCont.cuda(),val_coor.cuda()
                val_src, val_face,val_faceCont,  val_coor =Variable(val_src), Variable(val_face),Variable(val_faceCont),Variable(val_coor)
                gfeat, pfeat = model(val_src,val_face,val_faceCont,val_coor)
                gfeat = gfeat.data.cpu()
                pfeat = pfeat.data.cpu()
                for imgId,gfeat,pfeat,label in zip(val_ImgId,gfeat,pfeat,val_label):
                    gfeat_dic[imgId].append(gfeat)
                    pfeat_dic[imgId].append(pfeat)
                    label_dic[imgId].append(label)
            imgIDs = list(gfeat_dic.keys())
            assert len(imgIDs)==val_num,'error, the size of val set is not the same'
            prob=[]
            realLabel=[]
            for idx,ids in enumerate(tqdm(imgIDs)):
                gfeats = gfeat_dic[ids]
                pfeats = pfeat_dic[ids]
                labels = label_dic[ids]
                gfeats=torch.stack(gfeats).cuda()
                pfeats=torch.stack(pfeats).cuda()
                logits = model_fc(gfeats,pfeats)
                hat_label = F.softmax(logits,1)[:,1]
                hat_label=hat_label.data.cpu().numpy()
                prob.append(hat_label)
                realLabel.append(np.argmax(labels)+1)
                cmc, val_mAP = GetResults(copy.deepcopy(prob), realLabel)

            if val_mAP>bestResult:
                save_name =saveDir+'/best_'+model_name
                torch.save(model.state_dict(),save_name)
                bestResult = val_mAP
            
                ####test Set####
                # extract the feature first
                gfeat_dic = defaultdict(list)
                pfeat_dic= defaultdict(list)
                label_dic = defaultdict(list)
                for test_batch_idx, (test_src,test_face, test_faceCont,test_coor, test_label, test_ImgId) in enumerate(tqdm(test_loader)):
                    if args.cuda:
                        test_src,test_face,test_faceCont, test_coor= test_src.cuda(),test_face.cuda(),test_faceCont.cuda(),test_coor.cuda()
                    test_src, test_face,test_faceCont,  test_coor =Variable(test_src), Variable(test_face),Variable(test_faceCont),Variable(test_coor)
                    gfeat, pfeat = model(test_src,test_face,test_faceCont,test_coor)
                    gfeat = gfeat.data.cpu()
                    pfeat = pfeat.data.cpu()
                    for imgId,gfeat,pfeat,label in zip(test_ImgId,gfeat,pfeat,test_label):
                        gfeat_dic[imgId].append(gfeat)
                        pfeat_dic[imgId].append(pfeat)
                        label_dic[imgId].append(label)
                imgIDs = list(gfeat_dic.keys())
                assert len(imgIDs)==test_num,'error, the size of val set is not the same'
                prob=[]
                realLabel=[]
                for idx,ids in enumerate(tqdm(imgIDs)):
                    gfeats = gfeat_dic[ids]
                    pfeats = pfeat_dic[ids]
                    labels = label_dic[ids]
                    gfeats=torch.stack(gfeats).cuda()
                    pfeats=torch.stack(pfeats).cuda()
                    logits = model_fc(gfeats,pfeats)
                    hat_label = F.softmax(logits,1)[:,1]
                    hat_label=hat_label.data.cpu().numpy()

                    prob.append(hat_label)
                    realLabel.append(np.argmax(labels)+1)
                    cmc, test_mAP = GetResults(copy.deepcopy(prob), realLabel)

        return val_mAP,accuracy/(num+(1e-6)),bestResult,test_mAP

    return val_mAP,accuracy/(num+(1e-6)),bestResult,test_mAP

    
val_mAP = 0.0
test_mAP=0.0
train_acc = 0.0
bestResult=0
import time
for epoch in range(0, args.epochs):
    val_mAP ,train_acc,bestResult,test_mAP= train(epoch, val_mAP,bestResult,test_mAP)
    # adjust_learning_rate(optimizer, epoch)
    print("{} :optimizer lr".format(model_name))
    scheduler.step()

save_name =saveDir+'/final_'+model_name
torch.save(model.state_dict(),save_name)