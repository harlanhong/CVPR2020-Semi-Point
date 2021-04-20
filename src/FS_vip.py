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
# setup_seed(20)
seed = random.randint(1,10000)
print('=============seed ;{}==============='.format(seed))
setup_seed(seed)
from memory import *
from dataset import *
from preprocessor import *
from relationNetwork import *
import torch.autograd
from sampler import *
import math
import cv2
from tqdm import tqdm  
from util.recorder import *
import copy
import time
from GetMapAndCMC import *
from option import args
from util.DataProvider import DataProvider
from loss import *
from util.utility import *
import util.ramps as ramps
from torch.optim import *
import loss
import relationNetwork
args.cuda = not args.no_cuda and torch.cuda.is_available()

VIPDataset = SemiNetDataset(root=args.dataset_path,index=args.index_name)

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
train_transformer1 = T.Compose([
        T.Resize((256,256)),
        T.RandomResizedCrop((224),(0.8, 0.85)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer
        ])

train_transformer2 = T.Compose([
        # T.ToPILImage()
        T.Resize((256,256)),
        T.RandomResizedCrop((224), (0.8, 0.85)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer
        ])
# train_transformer = T.Compose([
#         T.RandomResizedCrop((224),(0.8, 0.85)),
#         T.RandomHorizontalFlip(), #以0.5的概率水平翻转
#         T.RandomVerticalFlip(), #以0.5的概率垂直翻转
#         T.RandomRotation(10), #在（-10， 10）范围内旋转
#         T.ColorJitter(0.05, 0.05, 0.05, 0.05), #HSV以及对比度变化
#         T.ToTensor(), 
# ])

train_transformer3 = T.Compose([
        # T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        normalizer
        ])
coor_transformer = T.Compose([T.Resize((224, 224)),
                                T.ToTensor()
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
unlable_train_set = VIPDataset.unlabel_train  
lable_train_set = VIPDataset.label_train  
val_set = VIPDataset.val   
test_set = VIPDataset.test    
num_unlabel_train = VIPDataset.num_unlabel_train  
num_label_train = VIPDataset.num_label_train  
val_num = VIPDataset.num_val
test_num = VIPDataset.num_test
unlabeled_trainset = TotalSampleDataset(root=args.dataset_path,index=args.index_name,transform1=train_transformer1,transform2=train_transformer2,transform3=train_transformer3,tag='unlabel_data')
label_loader = DataLoader(
   SemiNetUnlabelPreprocessor_v2(lable_train_set, isTrain=True, 
                     transform1=train_transformer1,transform2=train_transformer2,transform3=train_transformer3),
                     sampler = RelationNetRandomFaceSampler(data_source=lable_train_set,
                                                     num_instances=args.ni),
                     batch_size=args.sample_size*args.ni,num_workers=args.nw,
                     pin_memory=True)
unlabel_loader = DataLoader(unlabeled_trainset,
                     batch_size=1,num_workers=args.nw,collate_fn=unlabeled_trainset.collate_fn,
                     pin_memory=True)
val_loader = DataLoader(
    RelationNetCPreprocessor(val_set, isTrain=False,
                     transform1=val_transformer1, transform2=val_transformer1,transform3=val_transformer3),
                     sampler = RelationNetTestSampler(data_source=val_set),
                     batch_size=args.batch_size, num_workers=args.nw,
                     pin_memory=True)
test_loader = DataLoader(
    RelationNetCPreprocessor(test_set, isTrain=False,
                     transform1=val_transformer1, transform2=val_transformer1,transform3=val_transformer3,coor_transformer=coor_transformer),
                     sampler = RelationNetTestSampler(data_source=test_set),
                     batch_size=args.batch_size, num_workers=args.nw,
                     pin_memory=True)
def adjust_learning_rate(opt, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr * (0.5 ** (epoch // args.lr_decay))
    for param_group in opt.param_groups:
        param_group['lr'] = lr
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def load_param(source,dst):
    dict_train=source.state_dict()
    dict_fc = dst.state_dict().copy()
    pretrained_dict = {k: v for k, v in dict_train.items() if k in dict_fc}#filter out unnecessary keys 
    dict_fc.update(pretrained_dict)
    dst.load_state_dict(dict_fc)
    return dst

labelProvider = DataProvider(label_loader)
unlabelProvider = DataProvider(unlabel_loader)
#The path and name saved by the model
# save_name='MT_unlabeled-loss-for-labeled-data-without-gamma'
print(args.save_name)
saveDir='../models2/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
model_name=args.save_name+"_"+str(seed)+'.pkl'

print(model_name)
recorder = Recorder(model_name)
#load model
Encoder = POINT_encoder(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
Relation = POINT_relation(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
Classifier = classifier()
LP = getattr(relationNetwork, args.label_propagation )().cuda()

#add graph

best=0
if args.cuda:
    Encoder = Encoder.cuda()
    Relation = Relation.cuda()
    Classifier = Classifier.cuda()
    LP.cuda()
    # model_teacher.cuda()
global global_step
global_step=0

CEloss = nn.CrossEntropyLoss().cuda()
MSEloss = getattr(loss, args.consistency_loss)().cuda()

if args.pre_train:
    args.lr*=0.01
    dict_pretrain = torch.load(saveDir+'best_baseline_1b.pkl')
    dict_model = Encoder.state_dict().copy()
    pretrained_dict = {k: v for k, v in dict_pretrain.items() if k in dict_model}#filter out unnecessary keys 
    dict_model.update(pretrained_dict)
    Encoder.load_state_dict(dict_model)

    dict_model = Relation.state_dict().copy()
    pretrained_dict = {k: v for k, v in dict_pretrain.items() if k in dict_model}#filter out unnecessary keys 
    dict_model.update(pretrained_dict)
    Relation.load_state_dict(dict_model)

    dict_model = Classifier.state_dict().copy()
    pretrained_dict = {k: v for k, v in dict_pretrain.items() if k in dict_model}#filter out unnecessary keys 
    dict_model.update(pretrained_dict)
    Classifier.load_state_dict(dict_model)
    load_param(Encoder,ema_Encoder)
    load_param(Relation,ema_Relation)
    load_param(Classifier,ema_Classifier)

print('=========================Lr: {}=========================='.format(args.lr))
optimizer = optim.SGD(list(Encoder.parameters())+list(Relation.parameters())+list(Classifier.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
scheduler = lr_scheduler.StepLR(optimizer,step_size = args.lr_decay,gamma = 0.5)

#Training
def main(epoch, val_mAP,bestResult,test_mAP):
    accuracy=0
    num=0
    total_loss=0
    total_unloss=0
    total_mem = 0
    total_entropy_loss=0
    Encoder.train()
    Relation.train()
    Classifier.train()
    global global_step
    for batch_idx in tqdm(range(args.training_step)):
 
        src,face, faceCont,coor,target, ImgId = labelProvider.next()
        if args.cuda:
            src_var,face_var,faceCont_var,coor_var, target_var = src[0].cuda(),face[0].cuda(),faceCont[0].cuda(),coor[0].cuda(), target[0].cuda()
        optimizer.zero_grad()
        labeled_gfeat,labeled_rfeat = Encoder(src_var,face_var,faceCont_var,coor_var)
        labeled_Ifeat = []
        for i in range(int(labeled_gfeat.shape[0]/args.ni)):
            Ifeat = Relation(labeled_gfeat[i*args.ni:(i+1)*args.ni],labeled_rfeat[i*args.ni:(i+1)*args.ni])
            labeled_Ifeat.append(Ifeat)
        labeled_Ifeat = torch.cat(labeled_Ifeat)
        labeled_logits = Classifier(labeled_Ifeat)
        celoss = CEloss(labeled_logits,target_var)
        

        celoss.backward()
        optimizer.step()
        # total_mem+=enloss.item()
        total_loss+=celoss.item()
        # total_entropy_loss+=entropy_loss_u.item()
        prediction = torch.max(F.softmax(labeled_logits), 1)[1]
        pred_y = prediction.data.cpu().numpy().squeeze()
        target_y = target_var.data.cpu().numpy()
        accuracy = sum(pred_y == target_y)+accuracy
        num=target_var.size(0) +num
        global_step+=1

    train_acc = accuracy/(num+10e-9)
    print('Epoch: {} Time: {} label_Loss: {:.6f} Val mAP: {:.4f} Test mAP {:.4f} Train Acc: {:.4f}'.format(
                    epoch,time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),total_loss/args.training_step,val_mAP,test_mAP,train_acc))

    # print('Epoch: {} Time: {} label_Loss: {:.6f} consistency_loss: {:.6f} entropy_loss: {:.6f} Val mAP: {:.4f} Test mAP {:.4f} Train Acc: {:.4f}'.format(
    #             epoch,time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),total_loss/len(label_loader),total_unloss/len(label_loader),total_mem/len(label_loader), val_mAP,test_mAP,train_acc))
    recorder.update('label loss', total_loss/args.training_step, epoch)
    recorder.update('train acc', accuracy/(num+10e-9), epoch)
    if epoch%args.interval==0 and epoch!=0:
        with torch.no_grad():
            Encoder.eval()
            Relation.eval()
            Classifier.eval()
            #Assign test model parameters
            # extract the feature first
            gfeat_dic = defaultdict(list)
            pfeat_dic= defaultdict(list)
            label_dic = defaultdict(list)

            for val_batch_idx, (val_src,val_face, val_faceCont,val_coor, val_label, val_ImgId) in enumerate(tqdm(val_loader)):
                if args.cuda:
                    val_src,val_face,val_faceCont, val_coor= val_src.cuda(),val_face.cuda(),val_faceCont.cuda(),val_coor.cuda()
                val_src, val_face,val_faceCont,  val_coor =Variable(val_src), Variable(val_face),Variable(val_faceCont),Variable(val_coor)
                gfeat, pfeat = Encoder(val_src,val_face,val_faceCont,val_coor)
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
            val_accuracy,val_number=0,0
            val_loss = 0

            for idx,ids in enumerate(tqdm(imgIDs)):
                gfeats = gfeat_dic[ids]
                pfeats = pfeat_dic[ids]
                labels = label_dic[ids]
                gfeats=torch.stack(gfeats).cuda()
                pfeats=torch.stack(pfeats).cuda()
                labels = torch.stack(labels).cuda()

                feats = Relation(gfeats,pfeats)
                logits = Classifier(feats)
                celoss = CEloss(logits,labels)
                val_loss+=celoss.item()
                prelogits = F.softmax(logits)

                #compute classification acc
                prediction = torch.max(prelogits, 1)[1]
                pred_y = prediction.data.cpu().numpy().squeeze()
                target_y = labels.data.cpu().numpy()
                val_accuracy = sum(pred_y == target_y)+val_accuracy
                val_number=pfeats.shape[0] + val_number 

                hat_label = prelogits[:,1].contiguous()
                hat_label=hat_label.data.cpu().numpy()
                prob.append(hat_label)
                realLabel.append(np.argmax(labels.data.cpu().numpy())+1)
            cmc, val_mAP = GetResults(copy.deepcopy(prob), realLabel)
            val_acc = val_accuracy/val_number
            save_name =saveDir+'/final_'+model_name
            state = {'encoder':Encoder.state_dict(),'relation':Relation.state_dict(), 'classifier': Classifier.state_dict(), 'args':args,'seed':seed}
            torch.save(state,save_name)
            val_loss = val_loss/len(imgIDs)
            recorder.update('val mAP', val_mAP, epoch)
            recorder.update('val cls acc', val_acc, epoch)
            recorder.update('val celoss', val_loss,epoch)

            if val_mAP>bestResult:
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
                    gfeat, pfeat = Encoder(test_src,test_face,test_faceCont,test_coor)
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
                test_loss=0
                test_accuracy = 0
                test_number = 0
                for idx,ids in enumerate(tqdm(imgIDs)):
                    gfeats = gfeat_dic[ids]
                    pfeats = pfeat_dic[ids]
                    labels = label_dic[ids]
                    gfeats=torch.stack(gfeats).cuda()
                    pfeats=torch.stack(pfeats).cuda()
                    labels = torch.stack(labels).cuda()

                    feats = Relation(gfeats,pfeats)
                    logits = Classifier(feats)
                    celoss = CEloss(logits,labels)
                    test_loss+=celoss.item()
                    prelogits = F.softmax(logits)

                    #compute classificeation loss
                    prediction = torch.max(prelogits, 1)[1]
                    pred_y = prediction.data.cpu().numpy().squeeze()
                    target_y = labels.data.cpu().numpy()
                    test_accuracy = sum(pred_y == target_y)+test_accuracy
                    test_number=pfeats.shape[0] + test_number 

                    hat_label = prelogits[:,1].contiguous()
                    hat_label=hat_label.data.cpu().numpy()
                    prob.append(hat_label)
                    realLabel.append(np.argmax(labels.data.cpu().numpy())+1)
                cmc, test_mAP = GetResults(copy.deepcopy(prob), realLabel)
                test_loss = test_loss/len(imgIDs)
                test_acc = test_accuracy/test_number
                save_name =saveDir+'/best_'+model_name
                state = {'encoder':Encoder.state_dict(),'relation':Relation.state_dict(), 'classifier': Classifier.state_dict(), 'args':args,'seed':seed}
                torch.save(state,save_name)
                recorder.update('test mAP', test_mAP, epoch)
                recorder.update('test cls acc', test_acc, epoch)
                recorder.update('test celoss', test_loss, epoch)
    
    return val_mAP,accuracy/(num+(1e-6)),bestResult,test_mAP

def rampUp(epoch=0,gamma=0):
   
    gamma = epoch*args.slope
    if gamma>1:
        return 1
    return gamma

val_mAP = 0.0
test_mAP=0.0
train_acc = 0.0
bestResult=0
gamma=0
global_mAP=0.83
import time
for epoch in range(0, args.epochs):
    val_mAP ,train_acc,bestResult,test_mAP= main(epoch, val_mAP,bestResult,test_mAP)
    # adjust_learning_rate(optimizer, epoch)

    print("{} :optimizer lr, best mAP: {}".format(model_name,bestResult))
    scheduler.step()
    recorder.save()
save_name =saveDir+'/final_'+model_name
state = {'encoder':Encoder.state_dict(),'relation':Relation.state_dict(), 'classifier': Classifier.state_dict(), 'args':args,'seed':seed}

torch.save(state,save_name)