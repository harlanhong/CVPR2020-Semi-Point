import numpy as np
import pdb
import random
import matplotlib.pyplot as plt 
# from scipy import interpolate
import torch.nn.functional as F
import torch
from collections import defaultdict
import matplotlib as mlp
import cv2
import shutil
from matplotlib.pyplot import MultipleLocator
mlp.rcParams['axes.spines.right'] = False
mlp.rcParams['axes.spines.top'] = False
#统计不同方法pseudo labeled中vip的个数变化
def task1():
    sm = '../../exp/Ours_draw_6362.pkl.pkl.npy'
    lp = '../../exp/LP_vip_draw_1975.pkl.pkl.npy'
    mt = '../../exp/MT_vip_draw_2411.pkl.pkl.npy'
    x = []
    y = []
    sm = np.array(np.load(sm,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(len(sm)):
        x.append(sm[i][0])
        y.append(np.sum(np.max(np.array(sm[i][1]),1)>0.5)/len(sm[i][1]))
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='sm',color="blue",linewidth=2) 

    x = []
    y = []
    sm = np.array(np.load(lp,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(len(sm)):
        x.append(sm[i][0])
        y.append(np.sum(np.max(np.array(sm[i][1]),1)>0.5)/len(sm[i][1]))
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='LP',color="red",linewidth=2) 

    x = []
    y = []
    sm = np.array(np.load(mt,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(len(sm)):
        x.append(sm[i][0])
        y.append(np.sum(np.max(np.array(sm[i][1]),1)>0.5)/len(sm[i][1]))
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='MT',color="green",linewidth=2) 

    x = []
    y = []
    sm = np.array(np.load(mt,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(200):
        x.append(i)
        y.append(1)
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='Ours',color="cyan",linewidth=2) 
    plt.xlabel(r"Epochs")
    plt.ylabel(r"Percentage(%)")

    #显示图示  
    plt.legend() 
    #保存图  
    plt.savefig("labelling.jpg")  
    print('save')
#统计全为non-vip百分比
def task2():
    sm = '../../exp/Ours_draw_8089.pkl.pkl.npy'
    lp = '../../exp/LP_vip_draw_1975.pkl.pkl.npy'
    mt = '../../exp/MT_vip_rdraw_3214.pkl.pkl.npy'
    x = []
    y = []
    sm = np.array(np.load(sm,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(len(sm)):
        x.append(sm[i][0])
        y.append(np.sum(np.max(np.array(sm[i][1]),1)<0.5)/len(sm[i][1]))
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='sm',color="blue",linewidth=2) 

    x = []
    y = []
    sm = np.array(np.load(lp,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(len(sm)):
        x.append(sm[i][0])
        y.append(np.sum(np.max(np.array(sm[i][1]),1)<0.5)/len(sm[i][1]))
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='LP',color="red",linewidth=2) 

    x = []
    y = []
    sm = np.array(np.load(mt,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(len(sm)):
        x.append(sm[i][0])
        y.append(np.sum(np.max(np.array(sm[i][1]),1)<0.5)/len(sm[i][1]))
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='MT',color="green",linewidth=2) 

    x = []
    y = []
    sm = np.array(np.load(mt,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(200):
        x.append(i)
        y.append(0)
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='Ours',color="cyan",linewidth=2) 

    #显示图示  
    plt.legend() 
    #保存图  
    plt.savefig("aw.jpg")
    print('save')
#统计全为vip的百分比
def task3():
    sm = '../../exp/Ours_draw_8089.pkl.pkl.npy'
    lp = '../../exp/LP_vip_draw_1975.pkl.pkl.npy'
    mt = '../../exp/MT_vip_rdraw_3214.pkl.pkl.npy'
    x = []
    y = []
    sm = np.array(np.load(sm,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(len(sm)):
        x.append(sm[i][0])
        y.append(np.sum(np.min(np.array(sm[i][1]),1)>0.5)/len(sm[i][1]))
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='sm',color="blue",linewidth=2) 

    x = []
    y = []
    sm = np.array(np.load(lp,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(len(sm)):
        x.append(sm[i][0])
        y.append(np.sum(np.min(np.array(sm[i][1]),1)>0.5)/len(sm[i][1]))
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='LP',color="red",linewidth=2) 

    x = []
    y = []
    sm = np.array(np.load(mt,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(len(sm)):
        x.append(sm[i][0])
        y.append(np.sum(np.min(np.array(sm[i][1]),1)>0.5)/len(sm[i][1]))
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='MT',color="green",linewidth=2) 

    x = []
    y = []
    sm = np.array(np.load(mt,allow_pickle=True).tolist()['important score'])
    # pdb.set_trace()
    for i in range(200):
        x.append(i)
        y.append(0)
    # f = interpolate.interp1d(x, y, kind='cubic')
    # x = np.linspace(np.min(x), np.max(x), 50)
    # y = f(x)
    plt.plot(x,y,label='Ours',color="cyan",linewidth=2) 

    #显示图示  
    plt.legend() 
    #保存图  
    plt.savefig("aw.jpg")  
    print('save')

def computeEW(IS):
    score = torch.Tensor(IS)
    ISW = F.softmax(score)
    onehot = torch.zeros([1,ISW.shape[0]])
    onehot[0,0] = 1
    perfect_IW = F.softmax(onehot,1).view(-1,1)
    max_IAW = 1-torch.div((perfect_IW*torch.log(perfect_IW)).sum(),np.log(1/perfect_IW.shape[0]))
    # pdb.set_trace()
    EW = (1-torch.div((ISW*torch.log(ISW)).sum(),np.log(1/ISW.shape[0])))/max_IAW
    return EW.item()

#查看每一个ENCAA
def task4():
    def bbx(key):
        if not 'jpg' in key:
            img = cv2.imread('/data/fating/OriginalDataset/MSDatasetv2/images/'+key+'.jpg')
        else:
            img = cv2.imread('/data/fating/OriginalDataset/MSDatasetv2/images/'+key)

        k = 0
        for rect in data[key]:
            rect = rect[:4]
            pt1 = (int(rect[0]),int(rect[1]))
            pt2 = (int(rect[2]),int(rect[3]))
            cv2.rectangle(img, pt1, pt2,(0,255,0),3)
            cv2.putText(img, str(k), pt1, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100, 200, 200), 5)
            k+=1
        result = cv2.imwrite('test.jpg',img)
        print(dict_ours_is[key])
        print(dict_pl_is[key])
        print(dict_lp_is[key])
        print(dict_mt_is[key])
        print(data[key])
    pl = 'EMS_PL_ALL_IS.npy'
    lp = 'EMS_LP_ALL_IS.npy'
    mt = 'EMS_MT_ALL_IS.npy'
    ours = 'EMS_Ours_ALL_IS.npy'

    lp_ISs = np.array(np.load(lp,allow_pickle=True).tolist()['IS'])
    lp_IDs = np.array(np.load(lp,allow_pickle=True).tolist()['ID'])
    mt_ISs = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    mt_IDs = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    pl_ISs = np.array(np.load(pl,allow_pickle=True).tolist()['IS'])
    pl_IDs = np.array(np.load(pl,allow_pickle=True).tolist()['ID'])
    ours_ISs = np.array(np.load(ours,allow_pickle=True).tolist()['IS'])
    ours_IDs = np.array(np.load(ours,allow_pickle=True).tolist()['ID'])
    
    # dict_pl_ew = defaultdict(float)
    dict_pl_is = defaultdict(list)
    dict_mt_is = defaultdict(list)
    dict_lp_is = defaultdict(list)
    dict_ours_is = defaultdict(list)

    for IS,ID in zip(lp_ISs,lp_IDs):
        dict_lp_is[ID] = IS
    for IS,ID in zip(mt_ISs,mt_IDs):
        dict_mt_is[ID] = IS
    for IS,ID in zip(pl_ISs,pl_IDs):
        dict_pl_is[ID] = IS
    for IS,ID in zip(ours_ISs,ours_IDs):
        dict_ours_is[ID] = IS
    data = np.load('/data/fating/OriginalDataset/MSDatasetv2/data/MSexpand_DSFD.npy',allow_pickle=True).tolist()
    # data = np.load('/data/fating/OriginalDataset/NCAAv2/data/ENCAA.npy',allow_pickle=True).tolist()
    data = data['test']
    keys = list(data.keys())
    result = []
    for key in keys:
        ISlp = dict_lp_is[key]
        ISmt = dict_mt_is[key]
        ISpl = dict_pl_is[key]
        ISours = dict_ours_is[key]
        # pdb.set_trace()
        if np.array(data[key])[np.argmax(ISlp),4]!=1 and np.array(data[key])[np.argmax(ISmt),4]!=1 and np.array(data[key])[np.argmax(ISpl),4]!=1 and np.array(data[key])[np.argmax(ISours),4]!=1:
            result.append(key)
            print(key)
        # if np.sum(ISlp>0.5)>2 and np.sum(ISmt>0.5)>2 and np.sum(ISpl>0.5)>2 and np.max(ISours)>0.5 and np.array(data[key])[np.argmax(ISlp),4]!=1 and np.array(data[key])[np.argmax(ISmt),4]!=1 and np.array(data[key])[np.argmax(ISpl),4]!=1:
        #     result.append(key)
        #     print(key)
    np.save('result.npy',result)
        # dict_lp_is[ID] = IS
    # print(img)
    #bbx(')
   
   
    pdb.set_trace()
    print('0000')
def task5():
    ew_dict = defaultdict(int)
    mt = 'total_is_MT.npy'
    iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    for IS,ID in zip(iss,ids):
        
        ew = computeEW(IS)
        ew_dict[ID] = ew
    # ews = np.array(ews)
    keys = list(ew_dict.keys())
    pdb.set_trace()
    print('aaa')
#查看每一个epochvip人数
def task6():
    mt99 = '../../exp/LP_tnrankS_ISW_EW_0.99.pkl.pkl.npy'
    mt99 = np.array(np.load(mt99,allow_pickle=True).tolist()['pseudo important score'])
    y = []
    for iss in mt99:
        sum99 = 0
        for is_ in iss[1]:
            # pdb.set_trace()
            sum99+=(np.sum((np.array(is_)/np.max(np.array(is_)))>0.99))
            # pdb.set_trace()
        y.append(sum99/len(iss[1]))
    x = list(range(len(y)))
     # y = f(x)
    plt.plot(x,y,label='0.99',color="cyan",linewidth=2) 

    mt99 = '../../exp/LP_tnrankS_ISW_EW_0.9.pkl.pkl.npy'
    mt99 = np.array(np.load(mt99,allow_pickle=True).tolist()['pseudo important score'])
    y = []
    for iss in mt99:
        sum99 = 0
        for is_ in iss[1]:
            # pdb.set_trace()
            sum99+=(np.sum((np.array(is_)/np.max(np.array(is_)))>0.99))
            # pdb.set_trace()
        y.append(sum99/len(iss[1]))
    x = list(range(len(y)))
     # y = f(x)
    plt.plot(x,y,label='0.9',color="red",linewidth=2) 

    mt99 = '../../exp/LP_tnrankS_ISW_EW_0.95.pkl.pkl.npy'
    mt99 = np.array(np.load(mt99,allow_pickle=True).tolist()['pseudo important score'])
    y = []
    for iss in mt99:
        sum99 = 0
        for is_ in iss[1]:
            # pdb.set_trace()
            sum99+=(np.sum((np.array(is_)/np.max(np.array(is_)))>0.99))
            # pdb.set_trace()
        y.append(sum99/len(iss[1]))
    x = list(range(len(y)))
     # y = f(x)
    plt.plot(x,y,label='0.95',color="blue",linewidth=2) 

    #显示图示  
    plt.legend() 
    #保存图  
    plt.savefig("aw.jpg")  
    print('save')

#看柱状图
def task7():
    mt99 = '../../exp/LP_tnrankS_ISW_EW_0.99.pkl.pkl.npy'
    mt99 = np.array(np.load(mt99,allow_pickle=True).tolist()['pseudo important score'])
    x = []
    one = []
    two = []
    three = []
    error = []
    for i in range(len(mt99)):
        if i%10==0:
            one_image = 0
            two_image = 0
            three_image = 0
            error_image = 0
            for vipscore in mt99[i][1]:
                # pdb.set_trace()
                vipscore = vipscore/np.max(vipscore)
                if np.sum(vipscore>0.99) == 1:
                    one_image+=1
                if np.sum(vipscore>0.99) == 2:
                    two_image+=1
                if np.sum(vipscore>0.99) == 3:
                    three_image+=1
                if np.sum(vipscore>0.99) > 3:
                    error_image+=1
            one.append(one_image)
            two.append(two_image)
            three.append(three_image)
            error.append(error_image)
            x.append(i)
    bar_width=0.2
    plt.bar(x=np.arange(len(x))-bar_width,height=one, label='1-vip',
        color='steelblue', alpha=0.8, width=bar_width)
     # y = f(x)
    plt.bar(x=np.arange(len(x)),height=two,
        label='2-vip', color='indianred', alpha=0.8, width=bar_width)
    
    plt.bar(x=np.arange(len(x))+bar_width, height=three,
        label='3-vip', color='green', alpha=0.8, width=bar_width)
    plt.bar(x=np.arange(len(x))+2*bar_width,height=error,
        label='>3-vip', color='cyan', alpha=0.8, width=bar_width)
    # 为两条坐标轴设置名称
    plt.xlabel("epoch")
    plt.ylabel("vip num")
    # 显示图例
    plt.legend()
    #保存图  
    plt.savefig("aw.jpg")  
    print('save')

#不同方法的vip数柱状图，for unlabelled EMS
def task8():
    fontsize = 14

    threshold = 0.5
    threshold2 = 0.99 
    zero = []
    one = []
    two = []
    three = []
    error = []
    data = defaultdict(list)
    mt = 'pl_is.npy'
    iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    total_num = len(iss)
    zero_image = 0
    one_image = 0
    two_image = 0
    three_image = 0
    error_image = 0
    for vipscore in iss:
        # pdb.set_trace()
        # vipscore = vipscore/np.max(vipscore)
        if np.sum(vipscore>threshold) == 0:
            zero_image+=1
        elif np.sum(vipscore>threshold) == 1:
            one_image+=1
        elif np.sum(vipscore>threshold) == 2:
            two_image+=1
        elif np.sum(vipscore>threshold) == 3:
            three_image+=1
        elif np.sum(vipscore>threshold) > 3:
            error_image+=1
    zero.append(zero_image)
    one.append(one_image)
    two.append(two_image)
    three.append(three_image)
    error.append(error_image)

    data['PL'] = [zero_image,one_image,two_image,three_image,error_image]

    
    
    mt = 'lp_is.npy'
    iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    zero_image = 0
    one_image = 0
    two_image = 0
    three_image = 0
    error_image = 0
    for vipscore in iss:
        # pdb.set_trace()
        # vipscore = vipscore/np.max(vipscore)
        if np.sum(vipscore>threshold) == 0:
            zero_image+=1
        elif np.sum(vipscore>threshold) == 1:
            one_image+=1
        elif np.sum(vipscore>threshold) == 2:
            two_image+=1
        elif np.sum(vipscore>threshold) == 3:
            three_image+=1
        elif np.sum(vipscore>threshold) > 3:
            error_image+=1
    zero.append(zero_image)
    one.append(one_image)
    two.append(two_image)
    three.append(three_image)
    error.append(error_image)
    data['LP'] = [zero_image,one_image,two_image,three_image,error_image]

    mt = 'mt_is.npy'
    iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    total_num = len(iss)
    zero_image = 0
    one_image = 0
    two_image = 0
    three_image = 0
    error_image = 0
    for vipscore in iss:
        # pdb.set_trace()
        # vipscore = vipscore/np.max(vipscore)
        if np.sum(vipscore>threshold) == 0:
            zero_image+=1
        elif np.sum(vipscore>threshold) == 1:
            one_image+=1
        elif np.sum(vipscore>threshold) == 2:
            two_image+=1
        elif np.sum(vipscore>threshold) == 3:
            three_image+=1
        elif np.sum(vipscore>threshold) > 3:
            error_image+=1
    zero.append(zero_image)
    one.append(one_image)
    two.append(two_image)
    three.append(three_image)
    error.append(error_image)
    data['MT'] = [zero_image,one_image,two_image,three_image,error_image]

    # mt = 'sm_is.npy'
    # iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    # ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    # zero_image = 0
    # one_image = 0
    # two_image = 0
    # three_image = 0
    # error_image = 0
    # for vipscore in iss:
    #     # pdb.set_trace()
    #     # vipscore = vipscore/np.max(vipscore)
    #     if np.sum(vipscore>threshold) == 0:
    #         zero_image+=1
        
    #     elif np.sum(vipscore>threshold) == 1:
    #         one_image+=1
    #     elif np.sum(vipscore>threshold) == 2:
    #         two_image+=1
    #     elif np.sum(vipscore>threshold) == 3:
    #         three_image+=1
    #     elif np.sum(vipscore>threshold) > 3:
    #         error_image+=1
    # zero.append(zero_image)
    # one.append(one_image)
    # two.append(two_image)
    # three.append(three_image)
    # error.append(error_image)
    # data['SM'] = [zero_image,one_image,two_image,three_image,error_image]

    mt = 'ours_sm_is.npy'
    iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    zero_image = 0
    one_image = 0
    two_image = 0
    three_image = 0
    error_image = 0
    for vipscore in iss:
        # pdb.set_trace()
        vipscore = vipscore/np.max(vipscore)
        if np.sum(vipscore>threshold2) == 0:
            zero_image+=1
        elif np.sum(vipscore>threshold2) == 1:
            one_image+=1
        elif np.sum(vipscore>threshold2) == 2:
            two_image+=1
        elif np.sum(vipscore>threshold2) == 3:
            three_image+=1
        elif np.sum(vipscore>threshold2) > 3:
            error_image+=1
    zero.append(zero_image)
    one.append(one_image)
    two.append(two_image)
    three.append(three_image)
    error.append(error_image)
    data['Ours'] = [zero_image,one_image,two_image,three_image,error_image]
    np.save('data&script/numOfVIP-EMS.npy',data)

    x = ["PL","LP","MT","Ours"]
    bar_width=0.17
    p1=plt.bar(x=np.arange(len(x))-2*bar_width,height=zero, label='0 important people',
        facecolor = 'royalblue',edgecolor = 'white',width=bar_width)
    p2=plt.bar(x=np.arange(len(x))-bar_width,height=one, label='1 important people',
        facecolor = 'darkorange',edgecolor = 'white',width=bar_width)
     # y = f(x)
    p3=plt.bar(x=np.arange(len(x)),height=two,
        label='2 important people', color='darkgreen',edgecolor = 'white', width=bar_width)
    
    p4=plt.bar(x=np.arange(len(x))+bar_width, height=three,
        label='3 important people', color='gold', edgecolor = 'white', width=bar_width)
    p5=plt.bar(x=np.arange(len(x))+2*bar_width,height=error,
        label='> 3 important people', color='purple', edgecolor = 'white', width=bar_width)
    plt.xticks(np.arange(len(x)), x,fontsize=fontsize) #将横坐标用cell替换,fontsize用来调整字体的大小
    plt.yticks(fontsize=fontsize) #将横坐标用cell替换,fontsize用来调整字体的大小
    plt.legend(loc = 'upper center')

    #保存图  
    plt.savefig("numOfVIP-EMS.jpg")
    plt.savefig("numOfVIP-EMS.pdf")  
    print('save')

#不同方法的vip数柱状图，for unlabelled ncaa
def task9():
    fontsize = 14

    threshold = 0.5
    threshold2 = 0.99 
    zero = []
    one = []
    two = []
    three = []
    error = []
    data = defaultdict(list)
    mt = 'ncaa_pl_is.npy'
    iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    total_num = len(iss)
    zero_image = 0
    one_image = 0
    two_image = 0
    three_image = 0
    error_image = 0
    for vipscore in iss:
        # pdb.set_trace()
        # vipscore = vipscore/np.max(vipscore)
        if np.sum(vipscore>threshold) == 0:
            zero_image+=1
        elif np.sum(vipscore>threshold) == 1:
            one_image+=1
        elif np.sum(vipscore>threshold) == 2:
            two_image+=1
        elif np.sum(vipscore>threshold) == 3:
            three_image+=1
        elif np.sum(vipscore>threshold) > 3:
            error_image+=1
    zero.append(zero_image)
    one.append(one_image)
    two.append(two_image)
    three.append(three_image)
    error.append(error_image)
    data['PL'] = [zero_image,one_image,two_image,three_image,error_image]

   

    mt = 'ncaa_lp_is.npy'
    iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    zero_image = 0
    one_image = 0
    two_image = 0
    three_image = 0
    error_image = 0
    for vipscore in iss:
        # pdb.set_trace()
        # vipscore = vipscore/np.max(vipscore)
        if np.sum(vipscore>threshold) == 0:
            zero_image+=1
        elif np.sum(vipscore>threshold) == 1:
            one_image+=1
        elif np.sum(vipscore>threshold) == 2:
            two_image+=1
        elif np.sum(vipscore>threshold) == 3:
            three_image+=1
        elif np.sum(vipscore>threshold) > 3:
            error_image+=1
    zero.append(zero_image)
    one.append(one_image)
    two.append(two_image)
    three.append(three_image)
    error.append(error_image)
    data['LP'] = [zero_image,one_image,two_image,three_image,error_image]

    mt = 'ncaa_mt_is.npy'
    iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    total_num = len(iss)
    zero_image = 0
    one_image = 0
    two_image = 0
    three_image = 0
    error_image = 0
    for vipscore in iss:
        # pdb.set_trace()
        # vipscore = vipscore/np.max(vipscore)
        if np.sum(vipscore>threshold) == 0:
            zero_image+=1
        elif np.sum(vipscore>threshold) == 1:
            one_image+=1
        elif np.sum(vipscore>threshold) == 2:
            two_image+=1
        elif np.sum(vipscore>threshold) == 3:
            three_image+=1
        elif np.sum(vipscore>threshold) > 3:
            error_image+=1
    zero.append(zero_image)
    one.append(one_image)
    two.append(two_image)
    three.append(three_image)
    error.append(error_image)
    data['MT'] = [zero_image,one_image,two_image,three_image,error_image]


    
    mt = 'ncaa_ours_sm_is.npy'
    iss = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    ids = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    zero_image = 0
    one_image = 0
    two_image = 0
    three_image = 0
    error_image = 0
    for vipscore in iss:
        # pdb.set_trace()
        vipscore = vipscore/np.max(vipscore)
        if np.sum(vipscore>threshold2) == 0:
            zero_image+=1
        elif np.sum(vipscore>threshold2) == 1:
            one_image+=1
        elif np.sum(vipscore>threshold2) == 2:
            two_image+=1
        elif np.sum(vipscore>threshold2) == 3:
            three_image+=1
        elif np.sum(vipscore>threshold2) > 3:
            error_image+=1
    zero.append(zero_image)
    one.append(one_image)
    two.append(two_image)
    three.append(three_image)
    error.append(error_image)
    data['Ours'] = [zero_image,one_image,two_image,three_image,error_image]
    np.save('data&script/numOfVIP-NCAA.npy',data)

    x = ["PL","LP","MT","Ours"]
    bar_width=0.17
    p1=plt.bar(x=np.arange(len(x))-2*bar_width,height=zero, label='0 important people',
        facecolor = 'royalblue',edgecolor = 'white',width=bar_width)
    p2=plt.bar(x=np.arange(len(x))-bar_width,height=one, label='1 important people',
        facecolor = 'darkorange',edgecolor = 'white',width=bar_width)
     # y = f(x)
    p3=plt.bar(x=np.arange(len(x)),height=two,
        label='2 important people', color='darkgreen',edgecolor = 'white', width=bar_width)
    
    p4=plt.bar(x=np.arange(len(x))+bar_width, height=three,
        label='3 important people', color='gold', edgecolor = 'white', width=bar_width)
    p5=plt.bar(x=np.arange(len(x))+2*bar_width,height=error,
        label='> 3 important people', color='purple', edgecolor = 'white', width=bar_width)
    plt.xticks(np.arange(len(x)), x,fontsize=fontsize) #将横坐标用cell替换,fontsize用来调整字体的大小
    plt.yticks(fontsize=fontsize) #将横坐标用cell替换,fontsize用来调整字体的大小

    # plt.ylabel(r"Number of unlabelled images",fontsize=12)
    plt.legend(loc = 'upper center',bbox_to_anchor=(0.56,1),fontsize=fontsize-2.6)

    #保存图  
    plt.savefig("numOfVIP-ENCAA.jpg")
    plt.savefig("numOfVIP-ENCAA.pdf")  
    print('save')

#统计图片中的人数
def task10():
    dic_person = defaultdict(int)
    data = np.load('/data/fating/OriginalDataset/MSDatasetv2/data/MSexpand_DSFD.npy',allow_pickle=True).tolist()
    loc = 22
    keys = list(data.keys())
    for i in range(loc):
        dic_person[i] = 0
  
    # dic_person[34]=0

    for key in keys:
        dataset = data[key]
        ks = list(dataset.keys())
        for k in ks:
            d = dataset[k]
            if len(d)<=20:
                dic_person[len(d)]+=1
            else:
                dic_person[loc]+=1
            # pdb.set_trace()
            # print('adfd')
    print(dic_person[0],dic_person[1])
    x1 = list(dic_person.keys())
    y1 = []
    for key in x1:
        y1.append(dic_person[key])
    x1 = np.array(x1)
    y1 = np.array(y1)
    sortIndex = np.argsort(x1)
    x1 = x1[sortIndex]
    y1 = y1[sortIndex]
    dic_person = defaultdict(int)
    data = np.load('/data/fating/OriginalDataset/NCAAv2/data/ENCAA.npy',allow_pickle=True).tolist()

    keys = list(data.keys())
    for i in range(loc):
        dic_person[i] = 0

    for key in keys:
        dataset = data[key]
        ks = list(dataset.keys())
        for k in ks:
            d = dataset[k]
            
            if len(d)<=20:
                dic_person[len(d)]+=1
            else:
                dic_person[loc]+=1
            # pdb.set_trace()
            # print('adfd')
    
    x2 = list(dic_person.keys())
    y2 = []
    for key in x2:
        y2.append(dic_person[key])

    x2 = np.array(x2)
    y2 = np.array(y2)
    sortIndex = np.argsort(x2)
    x2 = x2[sortIndex]
    y2 = y2[sortIndex]
    
    # plt.figure(figsize=(7,4.5))
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                # wspace=None, hspace=0.7)
    
    # plt.subplot(211)
    a = plt.bar(np.arange(len(x1)),y1,color=['royalblue','darkorange','darkgreen', 'gold', 'purple'])
    plt.grid(axis="y",linestyle='--')
    plt.xticks([0,5,10,15,20,loc], [0,5,10,15,20,'>20'],fontsize=15) #将横坐标用cell替换,fontsize用来调整字体的大小2
    plt.yticks(fontsize=15) #将横坐标用cell替换,fontsize用来调整字体的大小

    plt.savefig("statistics-EMS.jpg")  
    plt.savefig("statistics-EMS.pdf") 
    plt.figure() 
    # plt.yticks([500,1000,1500], [500,1000,1500],fontsize=14) #将横坐标用cell替换,fontsize用来调整字体的大小

    # plt.subplot(212)
    a = plt.bar(np.arange(len(x2)),y2,color=['royalblue','darkorange','darkgreen', 'gold', 'purple'])
    plt.grid(axis="y",linestyle='--')
    plt.xticks([0,5,10,15,20,loc], [0,5,10,15,20,'>20'],fontsize=15) #将横坐标用cell替换,fontsize用来调整字体的大小
    plt.yticks(fontsize=15) #将横坐标用cell替换,fontsize用来调整字体的大小

    # plt.ylabel('number of images', fontsize=12) 
    # plt.xlabel('number of peoples', fontsize=12) 
    # plt.title('Images vs Peoples (ENCAA)') 
    #保存图  
    plt.savefig("statistics-ENCAA.jpg")  
    plt.savefig("statistics-ENCAA.pdf")  

#统计图片中的人数
def task101():
    dic_person = defaultdict(int)
    data = np.load('/data/fating/OriginalDataset/MSDatasetv2/data/MSexpand_DSFD.npy',allow_pickle=True).tolist()

    keys = list(data.keys())
    for i in range(22):
        dic_person[i] = 0
 
    # dic_person[34]=0

    for key in keys:
        dataset = data[key]
        ks = list(dataset.keys())
        for k in ks:
            d = dataset[k]
            if len(d)<=20:
                dic_person[len(d)]+=1
            else:
                dic_person[22]+=1
            # pdb.set_trace()
            # print('adfd')
    print(dic_person[0],dic_person[1])
    x1 = list(dic_person.keys())
    y1 = []
    for key in x1:
        y1.append(dic_person[key])
    x1 = np.array(x1)
    y1 = np.array(y1)
    sortIndex = np.argsort(x1)
    x1 = x1[sortIndex]
    y1 = y1[sortIndex]
    dic_person = defaultdict(int)
    data = np.load('/data/fating/OriginalDataset/NCAAv2/data/ENCAA.npy',allow_pickle=True).tolist()

    keys = list(data.keys())
    for i in range(22):
        dic_person[i] = 0


    for key in keys:
        dataset = data[key]
        ks = list(dataset.keys())
        for k in ks:
            d = dataset[k]
            
            if len(d)<=20:
                dic_person[len(d)]+=1
            else:
                dic_person[22]+=1
            # pdb.set_trace()
            # print('adfd')
    print(dic_person[28])
    dic_person[0]=0
    dic_person[1]=0
    x2 = list(dic_person.keys())
    y2 = []
    for key in x2:
        y2.append(dic_person[key])

    x2 = np.array(x2)
    y2 = np.array(y2)
    sortIndex = np.argsort(x2)
    x2 = x2[sortIndex]
    y2 = y2[sortIndex]

    plt.figure(figsize=(9,4))

    # plt.xticks(np.arange(len(x1)), x1)
    a = plt.bar(np.arange(len(x1)),y1,color=['royalblue','darkorange','darkgreen', 'gold', 'purple'])
    # a = plt.bar(np.arange(len(x1)),y1)
    # autolabel(a)
    
    dist = 28
    a = plt.bar(np.arange(len(x2))+dist,y2,color=['royalblue','darkorange','darkgreen', 'gold', 'purple'])
    # autolabel(a)
    
    plt.xticks([0,5,10,15,20,22]+[0+dist,5+dist,10+dist,15+dist,20+dist,22+dist], [0,5,10,15,20,'>23']+[0,5,10,15,20,'>23'],fontsize=12) #将横坐标用cell替换,fontsize用来调整字体的大小
    plt.yticks([500,1000,1500,2000,2500,3000,3500,4000],[500,1000,1500,2000,2500,3000,3500,4000],fontsize=12) #将横坐标用cell替换,fontsize用来调整字体的大小
    plt.grid(axis="y",linestyle='--')

    #保存图  
    plt.savefig("statistics.jpg")  
    plt.savefig("statistics.pdf")  

##折线图 (PL,MT,LP,Softmax,Ours/3) test EMS
def task12():
    fontsize = 14
   
    def func(ISs):
        result = []
        for iss in ISs:
            iss = iss.tolist()
            iss+=[0,0,0,0,0,0,0,0]
            iss = np.array(iss)
            iss = -np.sort(-iss)
            result.append(iss[:8])
        
        result = np.stack(result)
        result = -np.sort(-result,1)
        result = np.mean(result,0)
        return result
    test_pl = 'test_pl_is.npy'
    test_mt = 'test_mt_is.npy'
    test_lp = 'test_lp_is.npy'

    test_sm = 'test_sm_is.npy'
    test_ours_sm_rank_isw_ew = 'test_ours_sm_is.npy'

    test_pl = np.array(np.load(test_pl,allow_pickle=True).tolist()['IS'])
    test_mt = np.array(np.load(test_mt,allow_pickle=True).tolist()['IS'])
    test_lp = np.array(np.load(test_lp,allow_pickle=True).tolist()['IS'])

    test_sm = np.array(np.load(test_sm,allow_pickle=True).tolist()['IS'])
    test_ours_sm_rank_isw_ew = np.array(np.load(test_ours_sm_rank_isw_ew,allow_pickle=True).tolist()['IS'])
    
    # pdb.set_trace()
    test_pl = func(test_pl)
    test_mt = func(test_mt)
    test_lp = func(test_lp)

    test_sm = func(test_sm)
    
    test_ours_sm_rank_isw_ew = func(test_ours_sm_rank_isw_ew)
    x = list(range(len(test_mt)))
    # plt.grid(axis="y",linestyle='--')
    markersize=7
    linewidth=1
    data = defaultdict(list)
    data['PL']=test_pl.tolist()
    data['MT']=test_mt.tolist()
    data['LP']=test_lp.tolist()
    data['SM']=test_sm.tolist()
    data['Ours']=test_ours_sm_rank_isw_ew.tolist()
    np.save('data&script/ours_Top-EMS.npy',data)
    plt.plot(x, test_pl, marker='d',label='PL', markersize=9, color='dodgerblue')
    plt.plot(x, test_lp, marker='o',label='LP', markersize=9, color='darkviolet')
    plt.plot(x, test_mt, marker='>',label='MT', markersize=9, color='goldenrod')
    # plt.plot(x, test_sm, marker='v',label='Softmax', markersize=9, color='green')
    plt.plot(x, test_ours_sm_rank_isw_ew, marker='*', label='Ours',markersize=13, color='firebrick')
    # plt.plot(x,test_pl,'.-',label='PL',color="gold",marker='^',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_lp,'.-',label='LP',color="black",marker='>',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_mt,'.-',label='MT',color="firebrick",marker='s',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_sm,'.-',label='Softmax',color="dodgerblue",marker='d',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_ours_sm_rank_isw_ew,'.-',label='Ours',color="limegreen",marker='*',linewidth=linewidth,markersize=markersize) 
 
    
    x = ["Top-1","Top-2","Top-3","Top-4","Top-5","Top-6","Top-7","Top-8"]
    plt.xticks(np.arange(len(x)), x,fontsize=fontsize,rotation=30)
    plt.yticks([0.2,0.4,0.5,0.6,0.8], [0.2,0.4,0.5,0.6,0.8],fontsize=fontsize)

    # plt.ylabel(r"Importance score",fontsize=12)

    #显示图示  
    plt.legend()  
    plt.savefig("ours_Top-EMS.jpg")  

    plt.savefig("ours_Top-EMS.pdf")  
    print('save')
##折线图  (PL,MT,LP,Softmax,Ours/3) test NCAA
def task13():
    fontsize = 14
    def func(ISs):
        result = []
        for iss in ISs:
            iss = iss.tolist()
            iss+=[0,0,0,0,0,0,0,0]
            iss = np.array(iss)
            iss = -np.sort(-iss)
            result.append(iss[:8])
        
        result = np.stack(result)
        result = -np.sort(-result,1)
        result = np.mean(result,0)
        return result
    test_pl = 'test_ncaa_pl_is.npy'
    test_mt = 'test_ncaa_mt_is.npy'
    test_lp = 'test_ncaa_lp_is.npy'

    test_sm = 'test_ncaa_sm_is.npy'
    test_ours_sm_rank_isw_ew = 'test_ncaa_ours_sm_is.npy'

    test_pl = np.array(np.load(test_pl,allow_pickle=True).tolist()['IS'])
    test_mt = np.array(np.load(test_mt,allow_pickle=True).tolist()['IS'])
    test_lp = np.array(np.load(test_lp,allow_pickle=True).tolist()['IS'])

    test_sm = np.array(np.load(test_sm,allow_pickle=True).tolist()['IS'])
    
    test_ours_sm_rank_isw_ew = np.array(np.load(test_ours_sm_rank_isw_ew,allow_pickle=True).tolist()['IS'])
   
    # pdb.set_trace()
    test_pl = func(test_pl)
    test_mt = func(test_mt)
    test_lp = func(test_lp)

    test_sm = func(test_sm)
    
    test_ours_sm_rank_isw_ew = func(test_ours_sm_rank_isw_ew)
    x = list(range(len(test_mt)))
    # plt.grid(axis="y",linestyle='--')
    data = defaultdict(list)
    data['PL']=test_pl.tolist()
    data['MT']=test_mt.tolist()
    data['LP']=test_lp.tolist()
    data['SM']=test_sm.tolist()
    data['Ours']=test_ours_sm_rank_isw_ew.tolist()
    np.save('data&script/ours_Top-NCAA.npy',data)
    markersize=7
    linewidth=1
    plt.plot(x, test_pl, marker='d',label='PL', markersize=9, color='dodgerblue')
    plt.plot(x, test_lp, marker='o',label='LP', markersize=9, color='darkviolet')
    plt.plot(x, test_mt, marker='>',label='MT', markersize=9, color='goldenrod')
    # plt.plot(x, test_sm, marker='v',label='Softmax', markersize=9, color='green')
    plt.plot(x, test_ours_sm_rank_isw_ew, marker='*', label='Ours',markersize=13, color='firebrick')
    # plt.plot(x,test_pl,'.-',label='PL',color="gold",marker='^',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_lp,'.-',label='LP',color="black",marker='>',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_mt,'.-',label='MT',color="firebrick",marker='s',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_sm,'.-',label='Softmax',color="dodgerblue",marker='d',linewidth=linewidth,markersize=markersize) 
    # # plt.plot(x,test_ours_sm_rank,'.-',label='Softmax + RankS',color="purple",linewidth=2) 
    # plt.plot(x,test_ours_sm_rank_isw,'.-',label='Ours: RankS+ISW',color="darkgreen",linewidth=2) 
    # plt.plot(x,test_ours_sm_rank_isw,'.-',label='Softmax + RankS + ISW',color="green",linewidth=2) 
    # plt.plot(x,test_ours_sm_rank_isw_ew,'.-',label='Ours',color="limegreen",marker='*',linewidth=linewidth,markersize=markersize) 
    x = ["Top-1","Top-2","Top-3","Top-4","Top-5","Top-6","Top-7","Top-8"]
    plt.xticks(np.arange(len(x)), x,fontsize=fontsize,rotation=30)
    plt.yticks([0.2,0.4,0.5,0.6,0.8], [0.2,0.4,0.5,0.6,0.8],fontsize=fontsize)
    # plt.ylabel(r"Importance score",fontsize=12)

    #显示图示  
    plt.legend(fontsize=fontsize)  
    plt.savefig("ours_Top-ENCAA.jpg")  

    plt.savefig("ours_Top-ENCAA.pdf")  
    print('save')
##将两个折线图放一起
def task1213():
    def func(ISs):
        result = []
        for iss in ISs:
            iss = iss.tolist()
            iss+=[0,0,0,0,0,0,0,0]
            iss = np.array(iss)
            iss = -np.sort(-iss)
            result.append(iss[:8])
        
        result = np.stack(result)
        result = -np.sort(-result,1)
        result = np.mean(result,0)
        return result
    test_pl = 'test_ncaa_pl_is.npy'
    test_mt = 'test_ncaa_mt_is.npy'
    test_lp = 'test_ncaa_lp_is.npy'

    test_sm = 'test_ncaa_sm_is.npy'
    test_ours_sm_rank_isw_ew = 'test_ncaa_ours_sm_is.npy'

    test_pl = np.array(np.load(test_pl,allow_pickle=True).tolist()['IS'])
    test_mt = np.array(np.load(test_mt,allow_pickle=True).tolist()['IS'])
    test_lp = np.array(np.load(test_lp,allow_pickle=True).tolist()['IS'])

    test_sm = np.array(np.load(test_sm,allow_pickle=True).tolist()['IS'])
    
    test_ours_sm_rank_isw_ew = np.array(np.load(test_ours_sm_rank_isw_ew,allow_pickle=True).tolist()['IS'])
   
    # pdb.set_trace()
    ncaa_test_pl = func(test_pl)
    ncaa_test_mt = func(test_mt)
    ncaa_test_lp = func(test_lp)

    ncaa_test_sm = func(test_sm)
    
    ncaa_test_ours_sm_rank_isw_ew = func(test_ours_sm_rank_isw_ew)

    test_pl = 'test_pl_is.npy'
    test_mt = 'test_mt_is.npy'
    test_lp = 'test_lp_is.npy'

    test_sm = 'test_sm_is.npy'
    test_ours_sm_rank_isw_ew = 'test_ours_sm_is.npy'

    test_pl = np.array(np.load(test_pl,allow_pickle=True).tolist()['IS'])
    test_mt = np.array(np.load(test_mt,allow_pickle=True).tolist()['IS'])
    test_lp = np.array(np.load(test_lp,allow_pickle=True).tolist()['IS'])

    test_sm = np.array(np.load(test_sm,allow_pickle=True).tolist()['IS'])
    test_ours_sm_rank_isw_ew = np.array(np.load(test_ours_sm_rank_isw_ew,allow_pickle=True).tolist()['IS'])
    
    # pdb.set_trace()
    test_pl = func(test_pl)
    test_mt = func(test_mt)
    test_lp = func(test_lp)

    test_sm = func(test_sm)
    
    test_ours_sm_rank_isw_ew = func(test_ours_sm_rank_isw_ew)

    x = list(range(len(test_mt)))
    plt.figure(figsize=(9, 4))
    # plt.grid(axis="y",linestyle='--')
    
    markersize=7
    linewidth=1
    plt.plot(x, test_pl, marker='d',label='PL', markersize=9, color='dodgerblue')
    plt.plot(x, test_lp, marker='o',label='LP', markersize=9, color='darkviolet')
    plt.plot(x, test_mt, marker='>',label='MT', markersize=9, color='goldenrod')
    # plt.plot(x, test_sm, marker='v',label='Softmax', markersize=9, color='green')
    plt.plot(x, test_ours_sm_rank_isw_ew, marker='*', label='Ours',markersize=13, color='firebrick')

    x = np.array([8,9,10,11,12,13,14,15])+1
    plt.plot(x, ncaa_test_pl, marker='d', markersize=9, color='dodgerblue')
    plt.plot(x, ncaa_test_lp, marker='o',markersize=9, color='darkviolet')
    plt.plot(x, ncaa_test_mt, marker='>', markersize=9, color='goldenrod')
    # plt.plot(x, ncaa_test_sm, marker='v', markersize=9, color='green')
    plt.plot(x, ncaa_test_ours_sm_rank_isw_ew, marker='*', markersize=13, color='firebrick')
    # plt.plot(x,test_pl,'.-',label='PL',color="gold",marker='^',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_lp,'.-',label='LP',color="black",marker='>',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_mt,'.-',label='MT',color="firebrick",marker='s',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_sm,'.-',label='Softmax',color="dodgerblue",marker='d',linewidth=linewidth,markersize=markersize) 
    # # plt.plot(x,test_ours_sm_rank,'.-',label='Softmax + RankS',color="purple",linewidth=2) 
    # plt.plot(x,test_ours_sm_rank_isw,'.-',label='Ours: RankS+ISW',color="darkgreen",linewidth=2) 
    # plt.plot(x,test_ours_sm_rank_isw,'.-',label='Softmax + RankS + ISW',color="green",linewidth=2) 
    # plt.plot(x,test_ours_sm_rank_isw_ew,'.-',label='Ours',color="limegreen",marker='*',linewidth=linewidth,markersize=markersize) 
    x = ["Top1","Top2","Top3","Top4","Top5","Top6","Top7","Top8"]+["","Top1","Top2","Top3","Top4","Top5","Top6","Top7","Top8"]
    plt.xticks(np.arange(17), x,fontsize=12,rotation=30)
    plt.yticks([0.2,0.4,0.5,0.6,0.8], [0.2,0.4,0.5,0.6,0.8],fontsize=12)
    plt.ylabel(r"Importance score",fontsize=12)

    #显示图示  
    plt.legend()  
    plt.savefig("ours_Top.jpg")  

    plt.savefig("ours_Top.pdf")  
    print('save')


def task14():
    EMS_SM_cmc = 'EMS_SM_cmc.npy'
    EMS_LP_cmc = 'EMS_LP_cmc.npy'
    EMS_PL_cmc = 'EMS_PL_cmc.npy'
    EMS_MT_cmc = 'EMS_MT_cmc.npy'
    EMS_Ours_cmc = 'EMS_Ours_cmc.npy'
    EMS_SM_cmc = np.array(np.load(EMS_SM_cmc,allow_pickle=True).tolist())*100
    EMS_LP_cmc = np.array(np.load(EMS_LP_cmc,allow_pickle=True).tolist())*100
    EMS_PL_cmc = np.array(np.load(EMS_PL_cmc,allow_pickle=True).tolist())*100
    EMS_MT_cmc = np.array(np.load(EMS_MT_cmc,allow_pickle=True).tolist())*100
    EMS_Ours_cmc = np.array(np.load(EMS_Ours_cmc,allow_pickle=True).tolist())*100
    plt.grid(linestyle='-.')

    x = list(range(8))

    plt.plot(x, EMS_PL_cmc[:len(x)], marker='d',label='PL', markersize=9, color='dodgerblue')
    plt.plot(x, EMS_LP_cmc[:len(x)], marker='o',label='LP', markersize=9, color='darkviolet')
    plt.plot(x, EMS_MT_cmc[:len(x)], marker='>',label='MT', markersize=9, color='goldenrod')
    plt.plot(x, EMS_SM_cmc[:len(x)], marker='v',label='Softmax', markersize=9, color='green')
    plt.plot(x, EMS_Ours_cmc[:len(x)], marker='*', label='Ours',markersize=13, color='firebrick')
    plt.xticks(np.arange(len(x)), [1,2,3,4,5,6,7,8,],fontsize=12)

    # plt.ylabel(r"Matching Rate (%)",fontsize=12)
    # plt.xlabel(r"Rank",fontsize=12)

    plt.legend()  
    plt.savefig("CMC-EMS.jpg")  

    plt.savefig("CMC-EMS.pdf")  
    print('save')
def task15():
    NCAA_SM_cmc = 'NCAA_SM_cmc.npy'
    NCAA_LP_cmc = 'NCAA_LP_cmc.npy'
    NCAA_PL_cmc = 'NCAA_PL_cmc.npy'
    NCAA_MT_cmc = 'NCAA_MT_cmc.npy'
    NCAA_Ours_cmc = 'NCAA_Ours_cmc.npy'
    NCAA_SM_cmc = np.array(np.load(NCAA_SM_cmc,allow_pickle=True).tolist())*100
    NCAA_LP_cmc = np.array(np.load(NCAA_LP_cmc,allow_pickle=True).tolist())*100
    NCAA_PL_cmc = np.array(np.load(NCAA_PL_cmc,allow_pickle=True).tolist())*100
    NCAA_MT_cmc = np.array(np.load(NCAA_MT_cmc,allow_pickle=True).tolist())*100
    NCAA_Ours_cmc = np.array(np.load(NCAA_Ours_cmc,allow_pickle=True).tolist())*100
    plt.grid(linestyle='-.')
    x = list(range(8))
    plt.plot(x, NCAA_PL_cmc[:len(x)], marker='d',label='PL', markersize=9, color='dodgerblue')
    plt.plot(x, NCAA_LP_cmc[:len(x)], marker='o',label='LP', markersize=9, color='darkviolet')
    plt.plot(x, NCAA_MT_cmc[:len(x)], marker='>',label='MT', markersize=9, color='goldenrod')
    plt.plot(x, NCAA_SM_cmc[:len(x)], marker='v',label='Softmax', markersize=9, color='green')
    plt.plot(x, NCAA_Ours_cmc[:len(x)], marker='*', label='Ours',markersize=13, color='firebrick')
    plt.xticks(np.arange(len(x)), [1,2,3,4,5,6,7,8,],fontsize=12)

    # plt.ylabel(r"Matching Rate (%)",fontsize=12)
    # plt.xlabel(r"Rank",fontsize=12)

    plt.legend()  
    plt.savefig("CMC-NCAA.jpg")  

    plt.savefig("CMC-NCAA.pdf")  
    print('save')
import os
def task16():
    def mymovefile(srcfile,dstfile):
        if not os.path.isfile(srcfile):
            print("%s not exist!"%(srcfile))
        else:
            fpath,fname=os.path.split(dstfile)    #分离文件名和路径
            if not os.path.exists(fpath):
                os.makedirs(fpath)                #创建路径
            shutil.copy(srcfile,dstfile)          #移动文件
            print("move %s -> %s"%( srcfile,dstfile))
    #/Users/Harlan/onedrive/MSDatasetv2/images/09752.jpg
    result = np.load('result.npy')
    print(result)
    for re in result:
        # pdb.set_trace()
        # /Users/Harlan/onedrive/target/07892.jpg
        # if int(re)<2310:
        # target = '/Users/Harlan/onedrive/NCAAv2/images/'+re
        # dst = '/Users/Harlan/onedrive/targetNCAA/'+re

        target = '/Users/Harlan/onedrive/MSDatasetv2/images/'+re+'.jpg'
        dst = '/Users/Harlan/onedrive/target/'+re+'.jpg'
        mymovefile(target,dst)
task4()
# task16()

# plt.figure()
# task9()
# plt.figure()
# task10()
# task1213()

# task12()
# plt.figure()
# task13()
# task14()
# plt.figure()
# task15()
