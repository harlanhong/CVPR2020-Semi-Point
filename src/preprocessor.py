from __future__ import absolute_import
import os.path as osp
import pdb
from PIL import Image
import torch
import numpy as np
import cv2
from option import args
from collections import defaultdict
import torch
def randomRotation(image, random_angle,mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        return image.rotate(random_angle, mode)
class ExteroiPreprocessor(object):
    def __init__(self, dataset, isTrain=True, transform1=None, transform2=None,transform3=None):
        # super(Preprocessor, self).__init__()
        self.dataset = dataset
  
        self.transform1 = transform1
        self.isTrain = isTrain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        image,Imgid= self.dataset[index]
        srcImg = Image.open(image).convert('RGB')
        srcImg = np.asarray(srcImg)
        # if self.transform1 is not None:
        #     srcImg = self.transform1(srcImg)
        return srcImg,Imgid
#relation network
class RelationNetCPreprocessor(object):
    def __init__(self, dataset, isTrain=True,  transform1=None, transform2=None,transform3=None,coor_transformer=None):
        # super(Preprocessor, self).__init__()
        self.dataset = dataset
     
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.coor_transformer=coor_transformer
        self.isTrain = isTrain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):

        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        src,Face, FaceCont,Coor, label, ImgId = self.dataset[index]
        srcImg = Image.open(src).convert('RGB')
        FaceImg = Image.open(Face).convert('RGB')
        FaceContImg = Image.open(FaceCont).convert('RGB')
        Coor = Image.open(Coor).convert('L')
        Coor=np.array(Coor,dtype=np.float32)
        # img = Image.open(frame_dir).convert('RGB')
        if self.transform1 is not None:
            Face = self.transform1(FaceImg)
        if self.transform2 is not None:
            FaceCont = self.transform2(FaceContImg)
        if self.transform3 is not None:
            src = self.transform3(srcImg)
        
        return src,Face,FaceCont,Coor,label, ImgId
class TransformTTimes:
    def __init__(self, transform,K=args.K):
        self.transform = transform
        self.K=K
    def __call__(self, inp):
        out=[]
        for i in range(self.K):
            out1 = self.transform(inp)
            out.append(out1)
        return out

class SemiNetUnlabelPreprocessor_v2(object):
    def __init__(self, dataset, isTrain=True, transform1=None, transform2=None,transform3=None,coor_transformer=None):
        self.dataset = dataset
        self.angle_dic = defaultdict(int)
        self.transform1 = TransformTTimes(transform1,args.K)
        self.transform2 = TransformTTimes(transform2,args.K)
        self.transform3 = TransformTTimes(transform3,args.K)
        self.coor_transformer=TransformTTimes(coor_transformer,args.K)
        self.isTrain = isTrain
        self.currentID=-1
        self.currentAngle=-1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):

        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        src,Face, FaceCont,Coor, label, ImgId = self.dataset[index]
        # if self.currentID==ImgId:
        #     random_angle = self.currentAngle
        # else:
        #     self.currentID=ImgId
        #     random_angle = np.random.randint(1, 360)
        #     self.currentAngle=random_angle            
        srcImg = Image.open(src).convert('RGB')
        FaceImg = Image.open(Face).convert('RGB')
        FaceContImg = Image.open(FaceCont).convert('RGB')
        Coor = Image.open(Coor).convert('L')
        Coor=np.array(Coor,dtype=np.float32)

        if self.transform1 is not None:
            # FaceImg = randomRotation(FaceImg,random_angle)
            Face1 = self.transform1(FaceImg)
        if self.transform2 is not None:
            # FaceContImg = randomRotation(FaceContImg,random_angle)
            FaceConts1 = self.transform2(FaceContImg)
        if self.transform3 is not None:
            srcs1 = self.transform3(srcImg)
        k = len(srcs1)
        Coor=[Coor for _ in range(k)]
        label=[label for _ in range(k)]
        ImgId = [ImgId for _ in range(k)]
        return srcs1,Face1,FaceConts1,Coor,label,ImgId

class SemiNetUnlabelPreprocessor(object):
    def __init__(self, dataset, isTrain=True, transform1=None, transform2=None,transform3=None,coor_transformer=None):
        self.dataset = dataset
     
        self.transform1 = TransformTTimes(transform1,args.K)
        self.transform2 = TransformTTimes(transform2,args.K)
        self.transform3 = TransformTTimes(transform3,args.K)
        self.coor_transformer=TransformTTimes(coor_transformer,args.K)
        self.isTrain = isTrain
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):

        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        src,Face, FaceCont,Coor, label, ImgId = self.dataset[index]
        srcImg = Image.open(src).convert('RGB')
        FaceImg = Image.open(Face).convert('RGB')
        FaceContImg = Image.open(FaceCont).convert('RGB')
        Coor = Image.open(Coor).convert('L')
        Coor=np.array(Coor,dtype=np.float32)

        if self.transform1 is not None:
            Face1,Faces2 = self.transform1(FaceImg)
        if self.transform2 is not None:
            FaceConts1,FaceConts2 = self.transform2(FaceContImg)
        if self.transform3 is not None:
            srcs1,srcs2 = self.transform3(srcImg)

        return srcs1,Face1,FaceConts1,Coor,label, ImgId,srcs2,Faces2,FaceConts2,Coor,label, ImgId
