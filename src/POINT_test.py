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

from dataset import *
from GetMapAndCMC import *
# from sampler import RandomIdentitySampler
from preprocessor import *
from relationNetwork import *
from sampler import *

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

test_set = VIPDataset.test
test_num = VIPDataset.num_test

test_loader = DataLoader(
    RelationNetCPreprocessor(test_set, isTrain=False,
                     transform1=test_transformer,transform2=test_transformer,transform3=test_transformer),
                     sampler = RelationNetTestSampler(data_source=test_set),
                     batch_size=args.batch_size, num_workers=args.nw,
                     pin_memory=True)
print(type(test_set[0][0]))
relation=1
modelName = args.model
print(modelName)
def updataParameters(load_dict,d_model):
    model_dict = d_model.state_dict()
    new_list = list(d_model.state_dict().keys()) 
    #model_dict[new_list[0]]
    pretrained_dict = {k: v for k, v in load_dict.items() if k in model_dict}#filter out unnecessary keys 
    model_dict.update(pretrained_dict)
    d_model.load_state_dict(model_dict)
model_feat = POINT_Module(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
model_fc = POINT_Module_fc(num_classes=2, num_instances=args.ni,h=args.h,N=args.N)
# Load the pretrained model
if args.cuda:
    model_feat = model_feat.cuda()
    model_fc = model_fc.cuda()
model_name = modelName
dict_train = torch.load(model_name)
model_feat.load_state_dict(dict_train)
# dict_new = model_fc.state_dict()
# new_list = list(model_fc.state_dict().keys())
# trained_list = list(dict_train.keys())
# copyLength=len(new_list)
# for k in range(copyLength):
#     dict_new[new_list[k]] = dict_train[trained_list[-(copyLength-k)]]
# model_fc.load_state_dict(dict_new)
updataParameters(dict_train,model_fc)
# Extracting features
def extract_feature():
    model_feat.eval()
    model_fc.eval()
     # extract the feature first
    gfeat_dic = defaultdict(list)
    pfeat_dic= defaultdict(list)
    label_dic = defaultdict(list)
    for val_batch_idx, (val_src,val_face, val_faceCont,val_coor, val_label, val_ImgId) in enumerate(tqdm(test_loader)):
        if args.cuda:
            val_src,val_face,val_faceCont, val_coor= val_src.cuda(),val_face.cuda(),val_faceCont.cuda(),val_coor.cuda()
        val_src, val_face,val_faceCont,  val_coor =Variable(val_src), Variable(val_face),Variable(val_faceCont),Variable(val_coor)
        gfeat, pfeat = model_feat(val_src,val_face,val_faceCont,val_coor)
        gfeat = gfeat.data.cpu()
        pfeat = pfeat.data.cpu()
        for imgId,gfeat,pfeat,label in zip(val_ImgId,gfeat,pfeat,val_label):
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
        hat_label = model_fc(gfeats,pfeats)
        hat_label=hat_label.data.cpu().numpy()
        prob.append(hat_label)
        realLabel.append(np.argmax(labels)+1)
        cmc, val_mAP = GetResults(copy.deepcopy(prob), realLabel)
        print(cmc)
        print(val_mAP)
start = time.time()
extract_feature()
end= time.time()
print('cost time:',(end-start)/test_num)
    # test()
