from torch import nn
import torch
import torchvision
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import pdb
import numpy as np
from PIL import Image
import random
import math
# from relationNetworkModule import relation_network
# import cv2
# import gpytorch
from option import args
from torchvision import transforms
def addGaussianNoise(x,sigma=0.05):
    if random.random() < 0.5:
        return x
    return x+torch.zeros_like(x.data).normal_()*sigma

class POINT_1b(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(POINT_1b, self).__init__()
        self.num_instances = num_instances
      
        
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        # self.faceNet = torchvision.models.resnet50(pretrained=True)
        # self.faceNet = nn.Sequential(*list(self.faceNet.children())[:-2])  ##[?,7,7,2048]

        # self.faceConNet = torchvision.models.resnet50(pretrained=True)
        # self.faceConNet = nn.Sequential(*list(self.faceConNet.children())[:-2])  ##[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(4352,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]

        self.relu = nn.ReLU()
        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)
        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img

    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)
        w_global = self.relu(img_feat_1)   #[n,1]

        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n,1]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) #[n,n,1]        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def forward(self,src,face,faceCon,Coor,sample_num=8):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        del src
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        faceCon_ = self.imgNet(faceCon)
        del faceCon
        face_ = self.imgNet(face)
        batch_size,d1,d2,d3=face.size()
        del face
        coor=self.CoorConv(Coor)
        del Coor
        ROIConv=torch.cat((face_,faceCon_,coor),1)
        del face_,faceCon_,coor
        roifeat=self.ROIConv(ROIConv)
        del ROIConv
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        # x = torch.cat((face_,faceCon_,Coor_),1)    #[32,2304,7,7]
        if not self.training:
            return src_,roifeat
        for k in range(self.N):
            face_attention_1=[]
            for i in range(int(batch_size /sample_num)):
                roifeat_ = roifeat[i *sample_num:(i + 1) *sample_num]
                img_ = src_[i *sample_num:(i + 1) *sample_num]
                img_embedding = self.img_feature_embedding(roifeat_,img_)
                attention_feat=[]
                for j in range(self.h):
                    attention_ = self.attention_module_multi_head(roifeat_,
                                img_embedding,img_,self.linear_global[k*self.h+j],self.linear_q[k*self.h+j],self.linear_k[k*self.h+j],
                                self.linear_a[k*self.h+j],self.linear_v[k*self.h+j])
                    attention_feat.append(attention_)
                attention_feat = torch.cat(attention_feat,1)
                face_attention_1.append(attention_feat)
            feat = torch.cat(face_attention_1)
            feat = feat+roifeat
            roifeat=feat
          
        x = self.subspace(roifeat)
        del roifeat
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        return prelogits
class POINT_1b_fc(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(POINT_1b_fc, self).__init__()
        self.store_w=[]
        self.store_w_global=[]
        self.store_w_a=[]
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        

        linear_global = [nn.Linear(self.feat_dim,1) for i in range(self.h*self.N)]
        linear_q=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_k=[nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        linear_a = [nn.Linear(int(self.feat_dim/self.h),1) for i in range(self.h*self.N)]
        linear_v = [nn.Linear(self.feat_dim,int(self.feat_dim/self.h)) for i in range(self.h*self.N)]
        self.linear_global=nn.ModuleList(linear_global)
        self.linear_q=nn.ModuleList(linear_q)
        self.linear_k=nn.ModuleList(linear_k)
        self.linear_a=nn.ModuleList(linear_a)
        self.linear_v=nn.ModuleList(linear_v)

        self.subspace = nn.Linear(1024, 128)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, src,face,imgName='none'):
        self.imgName=imgName
        n, d1 = face.size()
        for i in range(self.N):
            img_embedding = self.img_feature_embedding(face, src)
            attention_feat=[]
            for j in range(self.h):
                attention_ = self.attention_module_multi_head(face,
                            img_embedding,src,self.linear_global[i*self.h+j],self.linear_q[i*self.h+j],self.linear_k[i*self.h+j],
                            self.linear_a[i*self.h+j],self.linear_v[i*self.h+j])
                attention_feat.append(attention_)
            attention_feat = torch.cat(attention_feat,1)
            feat=attention_feat+face
            face=feat
        x = self.subspace(face)
        x = self.feat_bn(x)
        prelogits = self.classifier(x)
        if not self.training:
            prelogits = self.SM(prelogits)
            self.hat_label = prelogits[:,1].contiguous()
            # self.hat_label = torch.transpose(self.SM(torch.transpose(hat_label.view(n,n),0,1)),0,1)
            return self.hat_label
            # return torch.mean(hat_label,0)

    def img_feature_embedding(self,roi_feat,img_feat):
        n,d = roi_feat.size()
        w_img=torch.add(roi_feat,img_feat.expand(n,d))
        return w_img
    def attention_module_multi_head(self, roi_feat, img_embedding,img_feat,linear_global,linear_q,linear_k,linear_a,linear_v):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        #Eq 5
        img_embedding = img_embedding.cuda()
        img_feat_1 = linear_global(img_embedding)

        w_global = self.relu(img_feat_1)   #[n,1]
        #################TODO#####################
        # self.store_w_global.append(w_global)
        # Eq 4
        q_data = linear_q(roi_feat)  # [num_rois, 1024/h]
        k_data = linear_k(roi_feat)  # [num_rois, 1024/h]
        aff=[]
        n,d = q_data.size()
     
        for i in range(k_data.size(0)):
            aff.append(torch.add(k_data[i].expand(n, d), q_data))
    
        w_a = torch.stack(aff)
        w_a = w_a.view(-1,w_a.size(2)) 
        w_a = linear_a(w_a)   #[n*n,1]    
        w_a = self.relu(w_a)  #[n*n,1]
        # self.store_w_a.append(w_a)
        w_a = w_a.view(q_data.size(0),-1,1) #[n,n]

        #Eq 3 Eq 2
        w_a_=[]
        for i in range(w_a.size(0)):
            w_a_.append(w_a[i]*w_global[i])
        w_a=torch.stack(w_a_) #[n,n]
        w_a = nn.Softmax(1)(w_a) 
        # self.store_w.append(w_a)
        
        #Eq1
        fa=linear_v(roi_feat)  #[n,1024/h]
        n,d = fa.size()
        fa = fa.unsqueeze(1)
        fa_expand=[fa for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = w_a*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum

    def getFeatureData(self):
        return self.store_w_global,self.store_w_a,self.store_w

class subRelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 1024):
        super(subRelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WP = nn.Linear(key_feature_dim, 1, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, pfeat, gfeat):
        n,d = pfeat.size()
        gEmbeding=torch.add(pfeat,gfeat.expand(n,d))
        eps_g = self.relu(self.WG(gEmbeding))

        N,_ = pfeat.shape
        w_k = self.WK(pfeat)
        w_q = self.WQ(pfeat)

        eps_p=[]
        n,d = w_k.size()
        eps_p = [torch.add(w_k[i].expand(n, d), w_q) for i in range(n)]
        eps_p = torch.stack(eps_p)
        eps_p = self.relu(self.WP(eps_p))
      
        eps_p = eps_p.view(n,n,1)
        eps_hat=[]
        eps_hat = [eps_p[i]*eps_g[i] for i in range(n)]
        eps_hat=torch.stack(eps_hat) #[n,n]

        eps = nn.Softmax(1)(eps_hat) #[n,n,1]  

        pInter = self.WV(pfeat)
        pInter = pInter.unsqueeze(1)
        fa_expand=[pInter for i in range(n)]
        fa_expand = torch.cat(fa_expand,1) 
        
        fr = eps*fa_expand
        fr_sum = torch.sum(fr,0) #do sum operation on column
        return fr_sum
class RelationModule(nn.Module):
    def __init__(self,n_relations = 4, appearance_feature_dim=1024, geo_feature_dim = 1024):
        super(RelationModule, self).__init__()
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        self.key_feature_dim = int(appearance_feature_dim/n_relations)
        for N in range(self.Nr):
            self.relation.append(subRelationUnit(appearance_feature_dim, self.key_feature_dim, geo_feature_dim))
    def forward(self,  pfeat, gfeat ):
        isFirst=True
        for N in range(self.Nr):
            if(isFirst):
                concat = self.relation[N](pfeat,gfeat)
                isFirst=False
            else:
                concat = torch.cat((concat, self.relation[N](pfeat, gfeat)), -1)
        return concat+pfeat


class classifier(nn.Module):
    def __init__(self,num_classes=2,*kwargs):
        super(classifier, self).__init__()
        self.subspace = nn.Linear(1024, 128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        self.feat_bn = nn.BatchNorm1d(128)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self,features):
        features = self.subspace(features)
        feat_bn = self.feat_bn(features)
        logits = self.classifier(feat_bn)
        return logits        
class POINT_encoder(nn.Module):
    def __init__(self,num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(POINT_encoder, self).__init__()
        self.num_instances = num_instances
        self.feat_dim = 1024
        
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        self.imgNet = torchvision.models.resnet50(pretrained=True)
        self.imgNet = nn.Sequential(*list(self.imgNet.children())[:-2])   #[?,7,7,2048]

        self.hidden=256
        self.ROIConv=torch.nn.Sequential()
        self.ROIConv.add_module('conv_1',torch.nn.Conv2d(4352,1024,kernel_size=3,stride=1,padding=1)) #[?,1024,7,7]
        self.ROIConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=3, stride=2))  # [?,1024,3,3]
        self.ROIConv.add_module('conv_2', torch.nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1))
        self.ROIConv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=3, stride=1))  # [?,256,1,1]
        self.ROIFc = torch.nn.Linear(256,self.feat_dim)
        self.imgConv = torch.nn.Sequential()
        self.imgConv.add_module('conv_1', torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1))
        self.imgConv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=7, stride=1))  # [?,112,112,10]
        self.imgFc = torch.nn.Linear(256, self.feat_dim)

        self.CoorConv = torch.nn.Sequential()
        self.CoorConv.add_module('conv_1',torch.nn.Conv2d(1,10,kernel_size=5,stride=1,padding=2)) # [?,224,224,10]
        self.CoorConv.add_module('maxpool_1',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,112,112,10]
        self.CoorConv.add_module('conv_2',torch.nn.Conv2d(10,32,kernel_size=5,stride=1,padding=2)) #[?,112,112,32]
        self.CoorConv.add_module('maxpool_',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,56,56,32]
        self.CoorConv.add_module('conv_3',torch.nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2)) #[?,56,56,64]
        self.CoorConv.add_module('maxpool_3',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,28,28,64]
        self.CoorConv.add_module('conv_4',torch.nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2)) #[?,28,28,128]
        self.CoorConv.add_module('maxpool_4',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,14,14,128]
        self.CoorConv.add_module('conv_5',torch.nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2)) #[?,14,14,256]
        self.CoorConv.add_module('maxpool_5',torch.nn.MaxPool2d(kernel_size=2,stride=2)) #[?,7,7,256]      
    def forward(self,src,face,faceCon,Coor,sample_num=8):
        #
        Coor =Coor.unsqueeze(1)
        src_ = self.imgNet(src)
        src_ = self.imgConv(src_)
        src_ = src_.view(-1, self.hidden)
        src_ = self.imgFc(src_)
        faceCon_ = self.imgNet(faceCon)
        face_ = self.imgNet(face)
        batch_size,d1,d2,d3=face.size()
        coor=self.CoorConv(Coor)
        ROIConv=torch.cat((face_,faceCon_,coor),1)
        roifeat=self.ROIConv(ROIConv)
        roifeat=roifeat.view(-1, self.hidden)
        roifeat = self.ROIFc(roifeat)
        return src_,roifeat
        
class POINT_relation(nn.Module):
    def __init__(self, num_classes=0, num_instances=1,h=8,N=1,**kwargs):
        super(POINT_relation, self).__init__()
        self.store_w=[]
        self.store_w_global=[]
        self.store_w_a=[]
        self.imgName = 'none'
        self.num_instances = num_instances
        
        self.feat_dim = 1024
        self.relu = nn.ReLU()
        self.SM = nn.Softmax()
        self.h=h
        self.N=N
        self.attention_dim=128
        self.alpha = 0.85
        
        self.relation = nn.ModuleList()
        self.geo_dim=1024

        for k in range(N):
            self.relation.append(RelationModule(h,self.feat_dim,self.geo_dim))
    def forward(self, src,face,imgName='none'):
        for idx in range(self.N):
            face = self.relation[idx](face,src)
        return face


class RelationNetwork(nn.Module):
    """Graph Construction Module"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,1,kernel_size=3,padding=1),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2*2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)            # max-pool without padding 
        self.m1 = nn.MaxPool2d(2, padding=1) # max-pool with padding

    def forward(self, x):
        
        x = x.view(-1,64,4,4)
        pdb.set_trace()
        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0),-1) 
        out = F.relu(self.fc3(out))
        out = self.fc4(out) # no relu

        out = out.view(out.size(0),-1) # bs*1

        return out




class LabelPropagation(nn.Module):
    """Label Propagation"""
    def __init__(self, feat_dim=1024,num_classes =2):
        super(LabelPropagation, self).__init__()
        self.feat_dim=feat_dim
        # self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        # self.h_dim, self.z_dim = args['h_dim'], args['z_dim']
        self.num_classes = num_classes
        # self.encoder = CNNEncoder(args)
        # self.relation = RelationNetwork()
        self.alpha = torch.tensor([0.99], requires_grad=False).cuda()
        # self.alpha = nn.Parameter(torch.tensor([0.99]).cuda(0), requires_grad=True)
    def cosine_distance(self,x, y):
        x_y = torch.mm(x, y.transpose(0,1))
        x_norm = torch.sum(x*x,1).sqrt()
        x_norm = x_norm.view(-1,1)
        y_norm = torch.sum(y*y,1).sqrt()
        y_norm = y_norm.view(-1,1)
        cosine_distance = torch.div(x_y, torch.mm(x_norm, y_norm.transpose(0,1)))
        return cosine_distance
    
    def self_distances(self,x,y):
        x_y = torch.mm(x, y.transpose(0,1))
        masks = torch.eye(x_y.shape[0])
        masks = (masks==0).type(torch.float).cuda()
        W = torch.mul(x_y,masks)
        masks = (W>0).type(torch.float).cuda()
        nonNeg_W=torch.mul(W,masks)
        return nonNeg_W
    def forward(self, labeled_inputs,labeled_target,unlabeled_inputs,unlabeled_targets=None):
        # init
        eps = np.finfo(float).eps

        # [support, s_labels, query, q_labels] = inputs
        num_classes = self.num_classes
        num_support = labeled_inputs.shape[0]
        num_queries = unlabeled_inputs.shape[0]
        
        # Step1: Embedding
        emb_all     = torch.cat((labeled_inputs,unlabeled_inputs), 0)
        # emb_all = self.encoder(inp).view(-1,self.feat_dim)
        N, d    = emb_all.shape[0], emb_all.shape[1]

        # Step2: Graph Construction

        emb1    = torch.unsqueeze(emb_all,1) # N*1*d
        emb2    = torch.unsqueeze(emb_all,0) # 1*N*d
        W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N
        W       = torch.exp(-W/(2*args.sigma*args.sigma))

        masks = torch.eye(W.shape[0])
        masks = (masks==0).type(torch.float).cuda()
        W = torch.mul(W,masks)

        ## keep top-k values
        topk, indices = torch.topk(W, 4)
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)
        mask = ((mask+torch.t(mask))>0).type(torch.float32)      # union, kNN graph
        #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
        W    = W*mask

        ## normalize
        D       = W.sum(0)   
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S       = D1*W*D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = labeled_target
        #to onehot
        ys = ys.view(-1,1)
        ys = torch.zeros(num_support,num_classes).cuda().scatter_(1,ys,1)
        if unlabeled_targets is None:
            yu = torch.zeros(num_queries, num_classes).cuda()
        else:
            yu = unlabeled_targets
        y  = torch.cat((ys,yu),0)
        Fk  = torch.matmul(torch.inverse(torch.eye(N).cuda()-self.alpha*S+eps), y)
        Fq = Fk[num_support:, :] # query predictions
        # pdb.set_trace()
        return Fq

