from __future__ import print_function

import argparse
import copy
# from vps import VPS
import pdb
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
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
setup_seed(5)
from dataset import *
from GetMapAndCMC import *
# from sampler import RandomIdentitySampler
from preprocessor import *
from relationNetwork import *
from sampler import *
import os
from option import args
args.cuda = not args.no_cuda and torch.cuda.is_available()

# VIPDataset = RelatinNetCMSIP(root=args.dataset_path,index=args.index_name)
VIPDataset = SemiNetDataset(root=args.dataset_path,index=args.index_name)

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

test_transformer = T.Compose([
        # T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
        normalizer
        ])

test_set = VIPDataset.unlabel_train + VIPDataset.test + VIPDataset.label_train +VIPDataset.val 
test_num = VIPDataset.num_unlabel_train +VIPDataset.num_label_train +VIPDataset.num_test+VIPDataset.num_val

test_loader = DataLoader(
    RelationNetCPreprocessor(test_set, isTrain=False,
                     transform1=test_transformer,transform2=test_transformer,transform3=test_transformer),
                     sampler = RelationNetTestSampler(data_source=test_set),
                     batch_size=args.batch_size, num_workers=args.nw,
                     pin_memory=True)
print(type(test_set[0][0]))
modelName = args.model
print(modelName)
def updataParameters(load_dict,d_model):
    model_dict = d_model.state_dict()
    new_list = list(d_model.state_dict().keys()) 
    #model_dict[new_list[0]]
    pretrained_dict = {k: v for k, v in load_dict.items() if k in model_dict}#filter out unnecessary keys 
    model_dict.update(pretrained_dict)
    d_model.load_state_dict(model_dict)
Encoder = POINT_encoder(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
Relation = POINT_relation(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
Classifier = classifier()
# Load the pretrained model
if args.cuda:
    Encoder.cuda()
    Relation.cuda()
    Classifier.cuda()
model_name = modelName
dict_train = torch.load(model_name)
keys = list(dict_train.keys())
if 'relation' in keys:
    updataParameters(dict_train['encoder'],Encoder)
    updataParameters(dict_train['relation'],Relation)
    updataParameters(dict_train['classifier'],Classifier)
else:
    updataParameters(dict_train['encoder'],Encoder)
    updataParameters(dict_train['encoder'],Relation)
    updataParameters(dict_train['classifier'],Classifier)

# Extracting features
dict_weight = defaultdict(list)
def extract_feature():
    total_EW = 0
    total_num = 0
    vip_num = 0
    two_image = 0
    one_image = 0
    three_image = 0

    total_IS = []
    error_image = 0
    ISs = []
    IDs = []
    with torch.no_grad():
        Encoder.eval()
        Relation.eval()
        Classifier.eval()
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

            onehot = torch.zeros([1,gfeats.shape[0]])
            onehot[0,0] = 1
            perfect_IW = F.softmax(onehot,1).view(-1,1)
            max_EW = 1-torch.div((perfect_IW*torch.log(perfect_IW)).sum(),np.log(1/perfect_IW.shape[0]))
            
            feats = Relation(gfeats,pfeats)
            logits = Classifier(feats)
            prelogits = F.softmax(logits)
            # print(vips)
            IS = []
            hat_label = prelogits[:,1].contiguous()
            vipscore = hat_label.cpu().numpy()
            total_IS.append(vipscore)
            IDs.append(ids)
            vipscore = vipscore/np.max(vipscore)

            if np.sum(vipscore>0.99) == 1:
                one_image+=1
            if np.sum(vipscore>0.99) == 2:
                two_image+=1
            if np.sum(vipscore>0.99) == 3:
                three_image+=1
            if np.sum(vipscore>0.99) > 3:
                error_image+=1
            vipscore = -np.sort(-vipscore)
            vipscore = vipscore.tolist()
            
            vipscore+=[0,0,0,0,0,0,0]
            IS = vipscore[:8]
        
            ISs.append(IS)

            ISW = F.softmax(hat_label).view(-1)
            EW = (1-torch.div((ISW*torch.log(ISW)).sum(),np.log(1/ISW.shape[0])))/max_EW

            total_EW+=EW.item()
            total_num+=1
            dict_weight[ids].append(hat_label)
            dict_weight[ids].append(ISW)
            dict_weight[ids].append(EW)
            hat_label=hat_label.data.cpu().numpy()
            prob.append(hat_label)
            realLabel.append(np.argmax(labels.data.cpu().numpy())+1)
        cmc, test_mAP = GetResults(copy.deepcopy(prob), realLabel)
        print(modelName)
        print(cmc,test_mAP)
        print(total_EW/total_num)
        print(vip_num,len(imgIDs))
        name = os.path.basename(model_name)
        strs = name.split('_')
        name = name[:-(len(strs[-1])+1)]
        np.save('util/'+name,ISs)
        np.save('util/'+args.test_savename+'_IS.npy',{'IS':total_IS,'ID':IDs})
        np.save('util/'+args.test_savename+'_cmc.npy',cmc)

        print(one_image,two_image,three_image,error_image,len(imgIDs))
start = time.time()
extract_feature()
end= time.time()
print('cost time:',(end-start)/test_num)
    # test()

np.save('weight',dict_weight)