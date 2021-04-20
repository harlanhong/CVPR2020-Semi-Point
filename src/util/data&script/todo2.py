import numpy as np
import pdb
import random
import matplotlib.pyplot as plt 
# from scipy import interpolate
import torch.nn.functional as F
import torch
from collections import defaultdict
import matplotlib as mlp
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

#查看每一个
def task4():
    mt = 'total_is_MT.npy'
    ori_mt = 'total_is_MT_orgin.npy'
    ISs = np.array(np.load(mt,allow_pickle=True).tolist()['IS'])
    IDs = np.array(np.load(mt,allow_pickle=True).tolist()['ID'])
    ori_ISs = np.array(np.load(ori_mt,allow_pickle=True).tolist()['IS'])
    ori_IDs = np.array(np.load(ori_mt,allow_pickle=True).tolist()['ID'])
     
    for IS,ID,ori_IS,ori_ID in zip(ISs,IDs,ori_ISs,ori_IDs):
        print(IS,ID)
        print(ori_IS,ori_ID)

        pdb.set_trace()
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
    threshold = 0.5
    threshold2 = 0.99 
    zero = []
    one = []
    two = []
    three = []
    error = []

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

   

    mt = 'sm_is.npy'
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

    x = ["PL","MT","LP","Softmax","Ours"]
    bar_width=0.17
    plt.bar(x=np.arange(len(x))-2*bar_width,height=zero, label='0',
        facecolor = 'royalblue',edgecolor = 'white',width=bar_width)
    plt.bar(x=np.arange(len(x))-bar_width,height=one, label='1',
        facecolor = 'darkorange',edgecolor = 'white',width=bar_width)
     # y = f(x)
    plt.bar(x=np.arange(len(x)),height=two,
        label='2', color='darkgreen',edgecolor = 'white', width=bar_width)
    
    plt.bar(x=np.arange(len(x))+bar_width, height=three,
        label='3', color='gold', edgecolor = 'white', width=bar_width)
    plt.bar(x=np.arange(len(x))+2*bar_width,height=error,
        label='> 3', color='purple', edgecolor = 'white', width=bar_width)
    plt.xticks(np.arange(len(x)), x,fontsize=12) #将横坐标用cell替换,fontsize用来调整字体的大小
    plt.ylabel(r"Number of unlabelled images",fontsize=12)
    plt.legend()

    #保存图  
    plt.savefig("numOfVIP-EMS.jpg")
    plt.savefig("numOfVIP-EMS.pdf")  
    print('save')

#不同方法的vip数柱状图，for unlabelled ncaa
def task9():
    threshold = 0.5
    threshold2 = 0.99 
    zero = []
    one = []
    two = []
    three = []
    error = []

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

   

    mt = 'ncaa_sm_is.npy'
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

    x = ["PL","MT","LP","Softmax","Ours"]
    bar_width=0.17
    p1=plt.bar(x=np.arange(len(x))-2*bar_width,height=zero, label='0',
        facecolor = 'royalblue',edgecolor = 'white',width=bar_width)
    p2=plt.bar(x=np.arange(len(x))-bar_width,height=one, label='1',
        facecolor = 'darkorange',edgecolor = 'white',width=bar_width)
     # y = f(x)
    p3=plt.bar(x=np.arange(len(x)),height=two,
        label='2', color='darkgreen',edgecolor = 'white', width=bar_width)
    
    p4=plt.bar(x=np.arange(len(x))+bar_width, height=three,
        label='3', color='gold', edgecolor = 'white', width=bar_width)
    p5=plt.bar(x=np.arange(len(x))+2*bar_width,height=error,
        label='> 3', color='purple', edgecolor = 'white', width=bar_width)
    plt.xticks(np.arange(len(x)), x,fontsize=12) #将横坐标用cell替换,fontsize用来调整字体的大小
    plt.ylabel(r"Number of unlabelled images",fontsize=12)
    plt.legend(loc = 'upper center',bbox_to_anchor=(1,1))

    #保存图  
    plt.savefig("numOfVIP-ENCAA.jpg")
    plt.savefig("numOfVIP-ENCAA.pdf")  
    print('save')

#统计图片中的人数
def task10():
    dic_person = defaultdict(int)
    data = np.load('/data/fating/OriginalDataset/MSDatasetv2/data/MSexpand_DSFD.npy',allow_pickle=True).tolist()

    keys = list(data.keys())
    for i in range(35):
        dic_person[i] = 0
    dic_person[31]=0
    dic_person[32]=0
    dic_person[33]=0
    # dic_person[34]=0

    for key in keys:
        dataset = data[key]
        ks = list(dataset.keys())
        for k in ks:
            d = dataset[k]
            if len(d)<=30:
                dic_person[len(d)]+=1
            else:
                dic_person[33]+=1
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
    for i in range(28):
        dic_person[i] = 0

    dic_person[26] = 0
    dic_person[27] = 0
    dic_person[28] = 0

    for key in keys:
        dataset = data[key]
        ks = list(dataset.keys())
        for k in ks:
            d = dataset[k]
            
            if len(d)<=25:
                dic_person[len(d)]+=1
            else:
                dic_person[28]+=1
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

    plt.figure(21)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.7)
    plt.subplot(211)
    # plt.xticks(np.arange(len(x1)), x1)
    a = plt.bar(np.arange(len(x1)),y1,color=['royalblue','darkorange','darkgreen', 'gold', 'purple'])
    # a = plt.bar(np.arange(len(x1)),y1)

    # autolabel(a)
    
    
    plt.ylabel('number of images', fontsize=12) 
    plt.xlabel('number of peoples', fontsize=12)
    plt.xticks([0,5,10,15,20,25,30,33], [0,5,10,15,20,25,30,'>30'],fontsize=12) #将横坐标用cell替换,fontsize用来调整字体的大小
    plt.title('Images vs Peoples (EMS)') 
    plt.subplot(212)
    # plt.xticks(np.arange(len(x2)), x2)
    a = plt.bar(np.arange(len(x2)),y2,color=['royalblue','darkorange','darkgreen', 'gold', 'purple'])
    # a = plt.bar(np.arange(len(x2)),y2)
    # autolabel(a)
  
    plt.ylabel('number of images', fontsize=12) 
    plt.xlabel('number of peoples', fontsize=12) 
    plt.xticks([0,5,10,15,20,25,28], [0,5,10,15,20,25,'>25'],fontsize=12) #将横坐标用cell替换,fontsize用来调整字体的大小

    plt.title('Images vs Peoples (ENCAA)') 
    #保存图  
    plt.savefig("statics.jpg")  
    plt.savefig("statics.pdf")  

##折线图 (PL,MT,LP,Softmax,Ours/3) test EMS
def task12():
   
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
    test_ours_sm_rank = func(test_ours_sm_rank)
    test_ours_sm_rank_isw = func(test_ours_sm_rank_isw)
    test_ours_sm_rank_isw_ew = func(test_ours_sm_rank_isw_ew)
    x = list(range(len(test_mt)))
    # plt.grid(axis="y",linestyle='--')
    markersize=7
    linewidth=1
    plt.plot(x,test_pl,'.-',label='PL',color="gold",marker='^',linewidth=linewidth,markersize=markersize) 
    plt.plot(x,test_mt,'.-',label='MT',color="firebrick",marker='s',linewidth=linewidth,markersize=markersize) 
    plt.plot(x,test_lp,'.-',label='LP',color="black",marker='>',linewidth=linewidth,markersize=markersize) 
    plt.plot(x,test_sm,'.-',label='Softmax',color="dodgerblue",marker='d',linewidth=linewidth,markersize=markersize) 
    
    plt.plot(x,test_ours_sm_rank_isw_ew,'.-',label='Ours',color="limegreen",marker='*',linewidth=linewidth,markersize=markersize) 
 
    
    # plt.plot(x,test_ours_sm_rank_isw_ew,'.-',label='Ours: RankS+ISW+EW',color="royalblue",linewidth=2) 
    x = ["Top1","Top2","Top3","Top4","Top5","Top6","Top7","Top8"]
    plt.xticks(np.arange(len(x)), x,fontsize=12)
    # plt.yticks([0.5], [0.5],fontsize=12)
    # plt.grid(axis="y",linestyle='--')
    plt.yticks([0.2,0.4,0.5,0.6,0.8], [0.2,0.4,0.5,0.6,0.8],fontsize=12)

    plt.ylabel(r"Importance score",fontsize=12)

    #显示图示  
    plt.legend()  
    plt.savefig("ours_Top-EMS.jpg")  

    plt.savefig("ours_Top-EMS.pdf")  
    print('save')
##折线图  (PL,MT,LP,Softmax,Ours/3) test NCAA
def task13():
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
    test_ours_sm_rank = np.array(np.load(test_ours_sm_rank,allow_pickle=True).tolist()['IS'])
    test_ours_sm_rank_isw = np.array(np.load(test_ours_sm_rank_isw,allow_pickle=True).tolist()['IS'])
    test_ours_sm_rank_isw_ew = np.array(np.load(test_ours_sm_rank_isw_ew,allow_pickle=True).tolist()['IS'])
   
    # pdb.set_trace()
    test_pl = func(test_pl)
    test_mt = func(test_mt)
    test_lp = func(test_lp)

    test_sm = func(test_sm)
    test_ours_sm_rank = func(test_ours_sm_rank)
    test_ours_sm_rank_isw = func(test_ours_sm_rank_isw)
    test_ours_sm_rank_isw_ew = func(test_ours_sm_rank_isw_ew)
    x = list(range(len(test_mt)))
    # plt.grid(axis="y",linestyle='--')
    markersize=7
    linewidth=1
    plt.plot(x,test_pl,'.-',label='PL',color="gold",marker='^',linewidth=linewidth,markersize=markersize) 
    plt.plot(x,test_mt,'.-',label='MT',color="firebrick",marker='s',linewidth=linewidth,markersize=markersize) 
    plt.plot(x,test_lp,'.-',label='LP',color="black",marker='>',linewidth=linewidth,markersize=markersize) 
    plt.plot(x,test_sm,'.-',label='Softmax',color="dodgerblue",marker='d',linewidth=linewidth,markersize=markersize) 
    # plt.plot(x,test_ours_sm_rank,'.-',label='Softmax + RankS',color="purple",linewidth=2) 
    # plt.plot(x,test_ours_sm_rank_isw,'.-',label='Ours: RankS+ISW',color="darkgreen",linewidth=2) 
    # plt.plot(x,test_ours_sm_rank_isw,'.-',label='Softmax + RankS + ISW',color="green",linewidth=2) 
    plt.plot(x,test_ours_sm_rank_isw_ew,'.-',label='Ours',color="limegreen",marker='*',linewidth=linewidth,markersize=markersize) 
    x = ["Top1","Top2","Top3","Top4","Top5","Top6","Top7","Top8"]
    plt.xticks(np.arange(len(x)), x,fontsize=12)
    plt.yticks([0.2,0.4,0.5,0.6,0.8], [0.2,0.4,0.5,0.6,0.8],fontsize=12)

    plt.ylabel(r"Importance score",fontsize=12)

    #显示图示  
    plt.legend()  
    plt.savefig("ours_Top-ENCAA.jpg")  

    plt.savefig("ours_Top-ENCAA.pdf")  
    print('save')

# task8()
# plt.figure()
# task9()
# plt.figure()
# task10()
task12()
plt.figure()
task13()