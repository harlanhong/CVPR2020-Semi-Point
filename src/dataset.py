from __future__ import print_function, absolute_import
import os.path as osp

from collections import defaultdict
from torchvision import transforms as T
import torch
import numpy as np
from PIL import Image
import pdb
from torch.utils.data import Dataset
from glob import glob
import random
import os
from preprocessor import *


# from .dataset import Dataset
# from ..utils.osutils import mkdir_if_missing
# from ..utils.serialization import write_json

class SemiNetDataset(Dataset): #img+face+faceCont+coor
    # url = 'http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip'
    # md5 = '1c2d9fc1cc800332567a0da25a1ce68c'

    def __init__(self, root,index='./MSindex.npy'):
        # processing Train
        # self.num_videos = 8
        if not os.path.exists(index):
            index = index.replace('/data/fating/','/home/share/fating/',1)
            root = root.replace('/data/fating/','/home/share/fating/',1)
        self.root = root
        index = np.load(index,allow_pickle=True).tolist();
        index_unlabel_train = list(index['unlabel_data'].keys())
        index_label_train = list(index['label_data'].keys())
        index_test = list(index['test'].keys())
        index_val = list(index['val'].keys())
        # A list of Image Folders
        self.unlabel_train = []
        self.num_unlabel_train = len(index_unlabel_train)
        for i in range(self.num_unlabel_train):
            ind = index_unlabel_train[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[-5]
                self.unlabel_train.append((src[j],Faces[j], FaceConts[j], Coor[j],int(Label_), ind))
            if NumFace == 1:
                self.unlabel_train.append((src[0],Faces[0], FaceConts[0],Coor[0], 0, ind))


        self.label_train = []
        self.num_label_train = len(index_label_train)
        for i in range(self.num_label_train):
            ind = index_label_train[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[-5]
                self.label_train.append((src[j],Faces[j], FaceConts[j], Coor[j],int(Label_), ind))
            if NumFace == 1:
                self.label_train.append((src[0],Faces[0], FaceConts[0],Coor[0], 0, ind))
        # testing data
        self.test = []
        self.num_test = len(index_test)
        for i in range(self.num_test):
            ind = index_test[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[-5]
                self.test.append((src[j],Faces[j], FaceConts[j],Coor[j], int(Label_), ind))
        

        # val data
        self.val = []
        self.num_val = len(index_val)
        for i in range(self.num_val):
            ind = index_val[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[-5]
                self.val.append((src[j],Faces[j], FaceConts[j],Coor[j], int(Label_), ind))

class RelatinNetCMSIP(Dataset): #img+face+faceCont+coor
    # url = 'http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip'
    # md5 = '1c2d9fc1cc800332567a0da25a1ce68c'

    def __init__(self, root,index='./MSindex.npy'):
        # processing Train
        # self.num_videos = 8
        if not os.path.exists(index):
            index = index.replace('/data/fating/','/home/share/fating/',1)
            root = root.replace('/data/fating/','/home/share/fating/',1)
        self.root = root
            
        index = np.load(index).tolist()
        self.index_train = index['label_data']
        index_test = index['test']
        index_val = index['val']
        # A list of Image Folders
        ImageFolderList = sorted(glob(root + '/Image_*'))
        


        # testing data
        self.test = []
        self.num_test = len(index_test)
        for i in range(self.num_test):
            ind = index_test[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[len(FaceName_) - 5]
                self.test.append((src[j],Faces[j], FaceConts[j],Coor[j], int(Label_), ind))
        

        # val data
        self.val = []
        self.num_val = len(index_val)
        for i in range(self.num_val):
            ind = index_val[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[len(FaceName_) - 5]
                self.val.append((src[j],Faces[j], FaceConts[j],Coor[j], int(Label_), ind))


    def support_query(self):
        
        self.num_support = int(len(self.index_train)/2)
        index_support = random.sample(self.index_train, self.num_support) 
        self.support = []
        # self.num_support = len(index_support)
        for i in range(self.num_support):
            ind = index_support[i]
            ImageFolder_ = self.root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[len(FaceName_)-5]
                self.support.append((src[j],Faces[j], FaceConts[j], Coor[j],int(Label_), ind))
            if NumFace == 1:
                self.support.append((src[0],Faces[0], FaceConts[0],Coor[0], 0, ind))

        index_query = list(set(self.index_train) ^ set(index_support))
        self.query = []
        self.num_query = len(index_query)
        for i in range(self.num_query):
            ind = index_query[i]
            ImageFolder_ = self.root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[len(FaceName_)-5]
                self.query.append((src[j],Faces[j], FaceConts[j], Coor[j],int(Label_), ind))
            if NumFace == 1:
                self.query.append((src[0],Faces[0], FaceConts[0],Coor[0], 0, ind))
        return self.support,self.num_support,self.query,self.num_query

class TotalSampleDataset(Dataset):
    def __init__(self, root,index='./MSindex.npy',transform1=None,transform2=None,transform3=None,tag='unlabel_data'):
         # processing Train
        # self.num_videos = 8
        if not os.path.exists(index):
            index = index.replace('/data/fating/','/home/share/fating/',1)
            root = root.replace('/data/fating/','/home/share/fating/',1)
        
        self.transform1=TransformTTimes(transform1,2)
        self.transform2=TransformTTimes(transform2,2)
        self.transform3=TransformTTimes(transform3,2)

        self.root = root

        index = np.load(index,allow_pickle=True).tolist();
        keys_sample = list(index[tag].keys())
        self.num_sample = len(keys_sample)
       
        self.sample_set = defaultdict(list)
        for i in range(self.num_sample):
            ind = keys_sample[i]
            ImageFolder_ = root+'/Image_'+str(ind)
            src = sorted(glob(ImageFolder_ + '/Image/Image*'))
            Faces = sorted(glob(ImageFolder_ + '/Face/Image*'))
            FaceConts = sorted(glob(ImageFolder_ + '/FaceCont/Image*'))
            Coor = sorted(glob(ImageFolder_ + '/Coordinate/Image*'))
            NumFace = len(Faces)
            for j in range(NumFace):
                FaceName_ = Faces[j]
                Label_ = FaceName_[-5]
                self.sample_set[ind].append((src[j],Faces[j], FaceConts[j], Coor[j],int(Label_), ind))
            if NumFace == 1:
                self.sample_set[ind].append((src[0],Faces[0], FaceConts[0],Coor[0], 0, ind))
        self.keys = list(self.sample_set.keys())
        self.num_sample = len(self.keys)
        # num_record = defaultdict(int)
        # for key in self.keys:
        #     num = len(self.sample_set[key])
        #     num_record[key]=num
        # pdb.set_trace()
        # np.save('unlabeled_num_record.npy',num_record)
        
    def __len__(self):
        return self.num_sample
    
    def __getitem__(self,item):
        key = self.keys[item]
        datas = self.sample_set[key]
        srcs1=[]
        faces1 = []
        faceConts1 = []
        coors1 = []
        labels1 = []
        idxs1 = []
        srcs2=[]
        faces2 = []
        faceConts2 = []
        coors2 = []
        labels2 = []
        idxs2 = []
        for (src,Face, FaceCont, Coor,label, ind) in datas:
            srcImg = Image.open(src).convert('RGB')
            FaceImg = Image.open(Face).convert('RGB')
            FaceContImg = Image.open(FaceCont).convert('RGB')
            Coor = Image.open(Coor).convert('L')
            # Coor=np.array(Coor,dtype=np.float32)
            Coor = T.ToTensor()(Coor)
            if self.transform1 is not None:
                # FaceImg = randomRotation(FaceImg,random_angle)
                Face1,Face2 = self.transform1(FaceImg)
            if self.transform2 is not None:
                # FaceContImg = randomRotation(FaceContImg,random_angle)
                FaceCont1,FaceCont2 = self.transform2(FaceContImg)
            if self.transform3 is not None:
                src1,src2 = self.transform3(srcImg)
            coors1.append(Coor)
            srcs1.append(src1)
            faces1.append(Face1)
            faceConts1.append(FaceCont1)
            labels1.append(label)
            idxs1.append(ind)
            coors2.append(Coor)
            srcs2.append(src2)
            faces2.append(Face2)
            faceConts2.append(FaceCont2)
            labels2.append(label)
            idxs2.append(ind)
        return [srcs1,srcs2],[faces1,faces2],[faceConts1,faceConts2],[coors1,coors2],[torch.Tensor(labels1),torch.Tensor(labels2)],[idxs1,idxs2]
    def collate_fn(self,batch):
        assert len(batch)==1
        srcs,faces,faceConts,coors,labels,idxs = batch[0]
        
        srcs0 = torch.stack(srcs[0])
        srcs1 = torch.stack(srcs[1])

        faces0 = torch.stack(faces[0])
        faces1 = torch.stack(faces[1])

        faceConts0 = torch.stack(faceConts[0])
        faceConts1 = torch.stack(faceConts[1])

        coors0 = torch.cat(coors[0])
        coors1 = torch.cat(coors[1])

        return  [srcs0,srcs1],[faces0,faces1],[faceConts0,faceConts1],[coors0,coors1],labels,idxs
