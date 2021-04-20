import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from option import args
import pdb
from torch.nn import init



class SortStrategy3CrossEntropyLoss(nn.Module):
    def __init__(self,gpu=True):
        super(SortStrategy3CrossEntropyLoss,self).__init__()
        self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,preLogits,Label):
        # prob = F.softmax(preLogits)
        gap = torch.max(Label,1)[0]
        idxs = torch.argsort(gap,dim=0,descending=True)
        end=int(len(idxs)/3)
        high_prob = torch.stack([preLogits[i] for i in idxs[:end]])
        high_label = torch.stack([Label[i] for i in idxs[:end]])
        Pseudolabels = torch.argmax(high_label,1)
        loss = self.lossF(high_prob,Pseudolabels)
        return loss
class SortDisStrategy3CrossEntropyLoss(nn.Module):
    def __init__(self,gpu=True):
        super(SortDisStrategy3CrossEntropyLoss,self).__init__()
        self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,preLogits,Label):
        # prob = F.softmax(preLogits)
        gap = torch.max(Label,1)[0]
        idxs = torch.argsort(gap,dim=0,descending=True)
        end=int(len(idxs)/3)
        high_prob = torch.stack([preLogits[i] for i in idxs[:end]])
        high_label = torch.stack([Label[i] for i in idxs[:end]])
        logits = F.log_softmax(high_prob)
        loss = -torch.sum(high_label*logits).div(logits.shape[0])
        return loss

class CrossEntropyRegurization(nn.Module):
    def __init__(self,gpu=True):
        super(CrossEntropyRegurization,self).__init__()
        
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,preLogits):
        logits = F.softmax(preLogits)
        loss = -torch.sum(logits*torch.log(logits))
        return loss

class Learn_mse_loss(nn.Module):
    def __init__(self,dim=1024):
        super(Learn_mse_loss,self).__init__()
        self.dim=dim
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_softmax):
        assert input_logits.size() == target_softmax.size()
        input_softmax = F.softmax(input_logits, dim=1)
        # target_softmax = F.softmax(target_logits, dim=1)
        loss = F.mse_loss(input_softmax, target_softmax)
        # loss = torch.mean(loss,1).mean()
        return loss

class Learn_global_mse_loss(nn.Module):
    def __init__(self,dim=1024):
        super(Learn_global_mse_loss,self).__init__()
        self.dim=dim
        self.w_fc = nn.Linear(self.dim, 1)

        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_logits,rfeat,gfeat,isLabel=True):
        
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)


        if isLabel:
            wf = F.sigmoid(self.w_fc(gfeat))  #relaxed w_ij a continuous variable in the range from 0 to 1
            target = torch.ones(wf.shape).cuda().detach()
            w_loss = F.mse_loss(wf,target)
            constrain_loss = loss.view(-1,1).mean()+w_loss
        else:
            with torch.no_grad():
                wf = F.sigmoid(self.w_fc(gfeat))  #relaxed w_ij a continuous variable in the range from 0 to 1
            constrain_loss = (loss.view(-1,1)*wf).mean()
        return constrain_loss

class Learn_uncertainty_global_mse_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(Learn_uncertainty_global_mse_loss,self).__init__()
        self.dim=dim
        self.w_fc = nn.Linear(self.dim, 1)
        self.ni=8
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_logits,rfeat,gfeat,isLabel=True):
        
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        vip_sm = target_softmax[:,1]
        ni_logits = vip_sm.view(-1,self.ni)
        ni_sm = F.softmax(ni_logits, dim=1)
        uncertenty =1-torch.div(torch.sum(ni_sm*torch.log(ni_sm),dim=1),np.log(1/self.ni))

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)
        uncertenty=uncertenty.view(-1,1)
        loss = torch.mul(loss.view(-1,self.ni),uncertenty)
        if isLabel:
            wf = F.sigmoid(self.w_fc(gfeat))  #relaxed w_ij a continuous variable in the range from 0 to 1
            target = torch.ones(wf.shape).cuda().detach()
            w_loss = F.mse_loss(wf,target)
            constrain_loss = loss.view(-1,1).mean()+w_loss
        else:
            with torch.no_grad():
                wf = F.sigmoid(self.w_fc(gfeat))  #relaxed w_ij a continuous variable in the range from 0 to 1
            constrain_loss = (loss.view(-1,1)*wf).mean()
        return constrain_loss

class Learn_uncertainty_mse_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(Learn_uncertainty_mse_loss,self).__init__()
        self.dim=dim
        self.ni=8
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_logits,uncertenty ):
        
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)
        uncertenty=uncertenty.view(-1,1)
        loss = torch.mul(loss.view(-1,self.ni),uncertenty).mean()
        return loss

class Learn_uncertainty_balance_mse_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(Learn_uncertainty_balance_mse_loss,self).__init__()
        self.dim=dim
        self.ni=8
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_logits,rfeat=None,gfeat=None,isLabel=True):
        assert input_logits.size() == target_logits.size()

        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        target_onehot = torch.argmax(target_softmax,1)
        classes = torch.unique(target_onehot)
        masks = [(target_onehot==cla).type(torch.float32)  for cla in classes]
        masks = torch.stack(masks)
        masks = masks/torch.sum(masks,1).view(-1,1)
        masks = torch.sum(masks,0)
        uncertainty =1-torch.div(torch.sum(target_softmax*torch.log(target_softmax),dim=1),np.log(1/target_softmax.shape[1]))

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)
        uncertainty=uncertainty.view(-1,1)
        loss = torch.mul(torch.mul(loss.view(-1,1),uncertainty),masks.view(-1,1)).mean()
        return loss

class Learn_uncertainty_balance_crossEntropy_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(Learn_uncertainty_balance_crossEntropy_loss,self).__init__()
        self.dim=dim
        self.ni=8
        self.lossF = nn.CrossEntropyLoss(reduce = False,size_average=False).cuda()
    def forward(self,input_logits, target_logits,rfeat=None,gfeat=None,isLabel=True):
        assert input_logits.size() == target_logits.size()

        # input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        #construct target
        
        target_score = target_softmax[:,1].view(-1,self.ni)
        target_vip  = torch.argmax(target_score,1)
        target = torch.zeros_like(target_score).cuda()
        for i in range(target_vip.shape[0]):
            target[i,target_vip[i]]=1
        
        target = target.view(-1).type(torch.long)
        classes = torch.unique(target)

        masks = [(target==cla).type(torch.float32)  for cla in classes]
        masks = torch.stack(masks)
        masks = masks/torch.sum(masks,1).view(-1,1)
        masks = torch.sum(masks,0)
        uncertainty =1-torch.div(torch.sum(target_softmax*torch.log(target_softmax),dim=1),np.log(1/target_softmax.shape[1]))

        loss = self.lossF(input_logits,target)
        # pdb.set_trace()
        # loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        # loss = torch.mean(loss,1)
        uncertainty=uncertainty.view(-1,1)
        # loss = torch.mul(loss.view(-1,1),uncertainty).mean()
        loss = torch.mul(torch.mul(loss.view(-1,1),uncertainty),masks.view(-1,1)).mean()
        return loss

class Learn_Prior_mse_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(Learn_Prior_mse_loss,self).__init__()
        self.dim=dim
        self.ni=8
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_softmax,pro):
        assert input_logits.size() == target_softmax.size()

        input_softmax = F.softmax(input_logits, dim=1)
        # target_softmax = F.softmax(target_logits, dim=1)

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)
        loss = torch.mul(loss.view(-1,1),pro).mean()
        # loss = torch.mul(torch.mul(loss.view(-1,1),uncertainty),masks.view(-1,1)).mean()
        return loss
class sum_Learn_Prior_mse_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(sum_Learn_Prior_mse_loss,self).__init__()
        self.dim=dim
        self.ni=8
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_softmax,pro):
        assert input_logits.size() == target_softmax.size()

        input_softmax = F.softmax(input_logits, dim=1)
        # target_softmax = F.softmax(target_logits, dim=1)

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)
        loss = torch.mul(loss.view(-1,1),pro).sum()
        # loss = torch.mul(torch.mul(loss.view(-1,1),uncertainty),masks.view(-1,1)).mean()
        return loss

class Learn_Prior_uncertainty_mse_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(Learn_Prior_uncertainty_mse_loss,self).__init__()
        self.dim=dim
        self.ni=8
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_softmax,pro,uncertainty):
        assert input_logits.size() == target_softmax.size()

        input_softmax = F.softmax(input_logits, dim=1)
        # target_softmax = F.softmax(target_logits, dim=1)

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)

        loss = torch.mul(loss.view(-1,1),pro.view(-1,1)).view(-1,args.ni)
        loss = torch.mul(loss,uncertainty).mean()
        # loss = torch.mul(torch.mul(loss.view(-1,1),uncertainty),masks.view(-1,1)).mean()
        return loss
class sum_Learn_Prior_uncertainty_mse_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(sum_Learn_Prior_uncertainty_mse_loss,self).__init__()
        self.dim=dim
        self.ni=8
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_softmax,pro,uncertainty):
        assert input_logits.size() == target_softmax.size()

        input_softmax = F.softmax(input_logits, dim=1)
        # target_softmax = F.softmax(target_logits, dim=1)

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)

        loss = torch.mul(loss.view(-1,1),pro.view(-1,1)).view(-1,args.ni)
        loss = torch.mul(loss,uncertainty).sum()
        # loss = torch.mul(torch.mul(loss.view(-1,1),uncertainty),masks.view(-1,1)).mean()
        return loss

class Entropy_loss(nn.Module):
    def __init__(self,ni=8):
        super(Entropy_loss,self).__init__()
        self.ni=ni
    def forward(self,logits):
        logits = F.softmax(logits)
        vip_logits = logits[:,1]
        per_logtis = vip_logits.view(-1,self.ni)
        prob = F.softmax(per_logtis, dim=1)
        loss = -torch.mul(prob,torch.log(prob)).mean(1).mean()
        return loss

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False).mean(1).mean()

def Top_softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    maxP = torch.max(target_logits,1)[0]
    input_logits = [input_logits[i] for i in range(input_logits.shape[0]) if maxP[i]>args.prob_threshold]
    if len(input_logits)==0:
        return 0
    input_logits = torch.stack(input_logits)
    target_logits = torch.stack([target_logits[i] for i in range(target_logits.shape[0]) if maxP[i]>args.prob_threshold])
    
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def mse_loss(input_softmax, target_softmax):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_softmax.size() == target_softmax.size()
    num_classes = input_softmax.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)
def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes


def one_vip_loss(input_logits,ni=8,num_classes=2):
    input_softmax = F.softmax(input_logits, dim=1)[:,1]
    input_softmax = input_softmax.view(-1,ni).sum(1)
    target = torch.ones_like(input_softmax.data).cuda()
    loss = torch.pow(torch.sub(input_softmax,target),2).mean()
    return loss
def compatibility_loss(input_logits,target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    loss = -torch.mul(target_softmax,torch.log(input_softmax)).mean(1).mean()
    return loss

def linear_rampup_w(current):
    """Linear rampup"""
    assert current >= 0
    if current*args.slope>50:
        return 50
    else:
        return current*args.slope

#=========================================
class Learn_global_mse_loss_test(nn.Module):
    def __init__(self,dim=1024):
        super(Learn_global_mse_loss_test,self).__init__()
        self.dim=dim
        self.w_fc = nn.Linear(self.dim, 1)

        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,gfeat,rfeat):
        wf = F.sigmoid(self.w_fc(gfeat))  #relaxed w_ij a continuous variable in the range from 0 to 1
        # print(wf)
        return wf


class Siminary_loss(nn.Module):
    def __init__(self,gpu=True, margin=0.5):
        super(Siminary_loss, self).__init__()
        self.margin = margin
    def distance_matrix(self,x,y):
        distances = x.pow(2).sum(1, keepdim=True) + y.pow(2).sum(1).unsqueeze(0) - 2 * torch.mm(x, y.transpose(0, 1))
        return distances
    def cosine_distance(self,x, y):
        x_y = torch.mm(x, y.transpose(0,1))
        x_norm = torch.sum(x*x,1).sqrt()
        x_norm = x_norm.view(-1,1)
        y_norm = torch.sum(y*y,1).sqrt()
        y_norm = y_norm.view(-1,1)
        cosine_distance = torch.div(x_y, torch.mm(x_norm, y_norm.transpose(0,1)))
        return cosine_distance
    def self_distance(self,x,y):
        emb1    = torch.unsqueeze(x,1) # N*1*d
        emb2    = torch.unsqueeze(y,0) # 1*N*d
        W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N
        W       = torch.exp(-W/2)
      
        # masks = torch.eye(W.shape[0])
        # masks = (masks==0).type(torch.float).cuda()
        # W = torch.mul(W,masks)
        return W
   
    def forward(self, features, target):

        # vips = [features[i] for i in range(len(target))  if target[i]==1]
        # nonvips = [features[i] for i in range(len(target))  if target[i]==0]
        # vips = torch.stack(vips)
        # nonvips = torch.stack(nonvips)
        #construct target
        target1 = target.view(-1,1)
        target2 = target.view(1,-1)
        target = (np.logical_xor(target1.cpu(),target2.cpu())==0).type(torch.float32).cuda()

        dist = self.self_distance(features,features)
        loss = torch.abs(dist-target).mean()

        return loss

class Learn_Siminary_loss(nn.Module):
    def __init__(self,gpu=True, feat_dim = 128,margin=0.5):
        super(Learn_Siminary_loss, self).__init__()
        self.margin = margin
        self.feat_dim=feat_dim
        self.subspace = nn.Linear(2*self.feat_dim,128)
        self.measurement = nn.Linear(128, 1)
        self.feat_bn = nn.BatchNorm1d(128)
        init.kaiming_normal(self.subspace.weight, mode='fan_out')
        init.constant(self.subspace.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)
   
    def forward(self, features, target=None):
        n,d = features.shape
        features_cat = [torch.cat((features[i].expand(n, d), features),1) for i in range(n)]
        features_cat = torch.cat(features_cat)
       
        feats = self.subspace(features_cat)
        feats = self.feat_bn(feats)
        score = F.sigmoid(self.measurement(feats)).view(n,n)
        if not self.training:
            return score
        target1 = target.view(-1,1)
        target2 = target.view(1,-1)
        target = (np.logical_xor(target1.cpu(),target2.cpu())==0).type(torch.float32).cuda()
        loss =  ((score-target)**2).mean()

        return loss

class latent_mse_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(latent_mse_loss,self).__init__()
        self.dim=dim
        self.ni=8
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_logits,rfeat=None,gfeat=None,isLabel=True):
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        n,d = input_logits.shape
        latent = F.softmax(input_logits[:,1].view(-1,self.ni),1).view(-1,1)

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)
        loss = torch.mul(loss,latent.view(-1)).view(-1,self.ni).sum(1).mean()
        
        return loss

    
class Learn_latent_VW_mse_loss(nn.Module):
    def __init__(self,dim=1024,ni=8):
        super(Learn_latent_VW_mse_loss,self).__init__()
        self.dim=dim
        self.ni=8
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_logits,rfeat,gfeat,isLabel=True):
        
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        vip_sm = target_softmax[:,1]
        ni_logits = vip_sm.view(-1,self.ni)
        ni_sm = F.softmax(ni_logits, dim=1)
        uncertenty =1-torch.div(torch.sum(ni_sm*torch.log(ni_sm),dim=1),np.log(1/self.ni))

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False)
        loss = torch.mean(loss,1)
        uncertenty=uncertenty.view(-1,1)
        loss = torch.mul(loss.view(-1,self.ni),uncertenty)

        latent = F.softmax(input_logits[:,1].view(-1,self.ni),1).view(-1,1)

        loss = torch.mul(loss.view(-1),latent.view(-1)).view(-1,self.ni).sum(1).mean()
        return loss


class FeatureDistanceLoass(nn.Module):
    def __init__(self,gpu=True):
        super(FeatureDistanceLoass, self).__init__()
        self.margin = torch.tensor([0.5], requires_grad=False).cuda()

    def euclidean_dist(self,x, y):
        # x: N x D
        # y: M x D
        n,d = x.shape
        m,d = y.shape
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return 1-F.sigmoid(torch.pow(x - y, 2).mean(2))
    def euclidean_dist01(self,x, y):
        # x: N x D
        # y: M x D
        n,d = x.shape
        m,d = y.shape
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return (1-F.sigmoid(torch.pow(x - y, 2).mean(2)))/0.5
    def guass_dist(self,x,y):
        emb1    = torch.unsqueeze(x,1) # N*1*d
        emb2    = torch.unsqueeze(y,0) # 1*N*d
        W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N
        W       = torch.exp(-W/(2*0.25))
        return W
    def guass_dist1(self,x,y):
        emb1    = torch.unsqueeze(x,1) # N*1*d
        emb2    = torch.unsqueeze(y,0) # 1*N*d
        W       = ((emb1-emb2)**2).mean(2)   # N*N*d -> N*N
        W       = torch.exp(-W/2)
        return W
   
    def cosine_distance(self,x, y):
        x_y = torch.mm(x, y.transpose(0,1))
        x_norm = torch.sum(x*x,1).sqrt()
        x_norm = x_norm.view(-1,1)
        y_norm = torch.sum(y*y,1).sqrt()
        y_norm = y_norm.view(-1,1)
        cosine_distance = torch.div(x_y, torch.mm(x_norm, y_norm.transpose(0,1)))
        return cosine_distance
    def forward(self, features, labels):
        vips = features[labels==1]
        nonvips = features[labels==0]
        min_similary_vips = torch.min(self.guass_dist1(vips,vips))
        min_similary_nonvips = torch.min(self.guass_dist1(nonvips,nonvips))
        max_similary_dist_vips_nonvips = torch.max(self.guass_dist1(vips,nonvips))
        loss = F.relu(self.margin - 1/4*(min_similary_vips+min_similary_nonvips)+max_similary_dist_vips_nonvips)
        
        return loss

class Learn_Id_mse_loss(nn.Module):
    def __init__(self,dim=1024):
        super(Learn_Id_mse_loss,self).__init__()
        self.dim=dim
        self.num_recorder = np.load('unlabeled_num_record.npy').tolist()
        # self.lossF = nn.CrossEntropyLoss().cuda()
    def forward(self,input_logits, target_logits,imgID=None):
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

        loss = F.mse_loss(input_softmax, target_softmax, reduce = False,size_average=False).mean(1).view(-1,args.ni)
        scale = []
        imgID = np.unique(imgID)
        for id in imgID:
            num = self.num_recorder[id]
            scale.append(np.min([1,args.ni/num]))
        scale = torch.Tensor(scale).cuda().view(-1,1)
        loss = torch.mul(loss,scale).mean(1).mean()
        # loss = torch.mean(loss,1).mean()
        return loss 

class Learn_Entropy_loss(nn.Module):
    def __init__(self,dim = 1024, ni = 8):
        super(Learn_Entropy_loss,self).__init__()
    def forward(self,input_logits):
        prob = F.softmax(input_logits,1)
        important_class = prob[:,1]
        loss = -(F.softmax(important_class)*F.log_softmax(important_class)).sum()
        return loss

class Learn_margin_loss(nn.Module):
    def __init__(self,dim = 1024, ni = 8):
        super(Learn_margin_loss,self).__init__()
    def forward(self,input_logits,prob):

        important_class = input_logits[:,1]
        loss = -(F.softmax(important_class)*F.log_softmax(important_class)).sum()
        return loss

