import numpy as np
import pdb
import random
# from collections import defaultdict
# labeled_train  = defaultdict(list)
# labeled_val  = defaultdict(list)

# index_name = '/data/fating/OriginalDataset/NCAAv2/data/ENCAA.npy'
# index = np.load(index_name,allow_pickle=True).tolist()
# index_unlabel_train = list(index['unlabel_data'].keys())
# index_label_train = list(index['label_data'].keys())
# index_test = list(index['test'].keys())
# index_val = list(index['val'].keys())

# labeled_data = index_label_train+index_val

# length = len(labeled_data)


# val = random.sample(labeled_data, int(length/4)) 
# rest_data = (list(set(labeled_data).difference(set(val))))
# train = random.sample(rest_data, int(length/2)) 
# rest_data = (list(set(rest_data).difference(set(train))))
# for item in val:
#     print(item)
#     if item in index_val:
#         labeled_val[item] = index['val'][item]
#     elif item in index_label_train:
#         labeled_val[item] = index['label_data'][item]
#     else:
#         print("=============erorr===============")
# for item in train:
#     print(item)

#     if item in index_val:
#         labeled_train[item] = index['val'][item]
#     elif item in index_label_train:
#         labeled_train[item] = index['label_data'][item]
#     else:
#         print("=============erorr===============")

# for item in rest_data:
#     print(item)

#     if item in index_val:
#         index['unlabel_data'][item] = index['val'][item]
#     elif item in index_label_train:
#         index['unlabel_data'][item] = index['label_data'][item]
#     else:
#         print("=============erorr===============")
# np.save('/data/fating/OriginalDataset/NCAAv2/data/ENCAA_2*.npy',{'unlabel_data':index['unlabel_data'],'label_data':labeled_train,'test':index['test'],'val':labeled_val})
# print(len(rest_data))
# print('End')

# pdb.set_trace()


import numpy as np  
import matplotlib.pyplot as plt  

# index_name = '../../exp/LP_top_totalpro_uncertainty_training_step_28.pkl.npy'
ours_lp = '../../exp/LP_rankS_ISW_EW_record_8127.pkl.pkl.npy'
lp = '../../exp/LP_vip_record_5218.pkl.pkl.npy'
ours_mt = '../../exp/MT_rankS_ISW_EW_record_3129.pkl.pkl.npy'
mt = '../../exp/MT_vip_record_4399.pkl.pkl.npy'
# pl = '../../exp/PL_record_6492.pkl.pkl.npy'

ours_lp = np.array(np.load(ours_lp,allow_pickle=True).tolist()['important score'])
lp = np.array(np.load(lp,allow_pickle=True).tolist()['important score'])
ours_mt = np.array(np.load(ours_mt,allow_pickle=True).tolist()['important score'])
mt = np.array(np.load(mt,allow_pickle=True).tolist()['important score'])
# pl = np.array(np.load(pl,allow_pickle=True).tolist()['important score'])


# important!!!!!!!!!!
# x = []
# y = []
# for i in range(len(ours_mt)):
#     x.append(ours_lp[i][0])
#     y.append(np.max(np.array(ours_lp[i][1]),1).mean())
# plt.plot(x,y,label='Ours&LP',color="blue",linewidth=2) 
# x = []
# y = []
# for i in range(len(ours_mt)):
#     x.append(lp[i][0])
#     y.append(np.max(np.array(lp[i][1]),1).mean())
# plt.plot(x,y,label='LP',color="green",linewidth=2) 
# x = []
# y = []
# for i in range(len(ours_mt)):
#     x.append(ours_mt[i][0])
#     y.append(np.max(np.array(ours_mt[i][1]),1).mean())
# plt.plot(x,y,label='Ours&MT',color="red",linewidth=2) 
# x = []
# y = []
# for i in range(len(ours_mt)):
#     x.append(mt[i][0])
#     y.append(np.max(np.array(mt[i][1]),1).mean())
# plt.plot(x,y,label='MT',color="cyan",linewidth=2) 
# plt.xlabel('epochs')
# plt.ylabel('mean important score of vip')

# result = np.array(ours_lp[120][1])
# result = -np.sort(-result,1)
# y = np.mean(result,0)
# x = list(range(len(y)))
# plt.plot(x,y,'ko-',label='Ours&LP',color="blue",linewidth=2) 

# std = np.std(result,0)
# r1 = list(map(lambda x: x[0]-x[1], zip(y, std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(y, std)))
# plt.fill_between(x, r1, r2, color="blue", alpha=0.2)

# result = np.array(lp[120][1])
# result = -np.sort(-result,1)
# y = np.mean(result,0)
# x = list(range(len(y)))
# plt.plot(x,y,'k^-',label='LP',color="green",linewidth=2) 
# std = np.std(result,0)
# r1 = list(map(lambda x: x[0]-x[1], zip(y, std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(y, std)))
# plt.fill_between(x, r1, r2, color="green", alpha=0.2)

# result = np.array(ours_mt[120][1])
# result = -np.sort(-result,1)
# y = np.mean(result,0)
# x = list(range(len(y)))
# plt.plot(x,y,'ks-',label='Ours&MT',color="red",linewidth=2) 
# std = np.std(result,0)
# r1 = list(map(lambda x: x[0]-x[1], zip(y, std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(y, std)))
# plt.fill_between(x, r1, r2, color="red", alpha=0.2)

# result = np.array(mt[120][1])
# result = -np.sort(-result,1)
# y = np.mean(result,0)
# x = list(range(len(y)))
# plt.plot(x,y,'k*-',label='MT',color="cyan",linewidth=2) 
# std = np.std(result,0)
# r1 = list(map(lambda x: x[0]-x[1], zip(y, std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(y, std)))
# plt.fill_between(x, r1, r2, color="cyan", alpha=0.2)


rankS = 'best_MT_vip.npy'
rankS = np.array(np.load(rankS,allow_pickle=True).tolist())
# ranks = [a.cpu().numpy() for a in rankS]
result = np.array(rankS)
# pdb.set_trace()
result = -np.sort(-result,1)
y = np.mean(result,0)
x = list(range(len(y)))
plt.plot(x,y,'ko-',label='MT',color="blue",linewidth=2) 

# std = np.std(result,0)
# r1 = list(map(lambda x: x[0]-x[1], zip(y, std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(y, std)))
# plt.fill_between(x, r1, r2, color="blue", alpha=0.2)


rankS = 'best_MT_rankS.npy'
rankS = np.array(np.load(rankS,allow_pickle=True).tolist())
# ranks = [a.cpu().numpy() for a in rankS]
result = np.array(rankS)
# pdb.set_trace()
result = -np.sort(-result,1)
y = np.mean(result,0)
x = list(range(len(y)))
plt.plot(x,y,'k^-',label='MT+rankS',color="green",linewidth=2) 
# std = np.std(result,0)
# r1 = list(map(lambda x: x[0]-x[1], zip(y, std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(y, std)))
# plt.fill_between(x, r1, r2, color="green", alpha=0.2)

rankS = 'best_MT_rankS_ISW.npy'
rankS = np.array(np.load(rankS,allow_pickle=True).tolist())
# ranks = [a.cpu().numpy() for a in rankS]

result = np.array(rankS)
# pdb.set_trace()
result = -np.sort(-result,1)
y = np.mean(result,0)
x = list(range(len(y)))
plt.plot(x,y,'k^-',label='MT+rankS+ISW',color="cyan",linewidth=2) 
# std = np.std(result,0)
# r1 = list(map(lambda x: x[0]-x[1], zip(y, std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(y, std)))
# plt.fill_between(x, r1, r2, color="cyan", alpha=0.2)

rankS = 'best_MT_rankS_ISW_EW_record.npy'
rankS = np.array(np.load(rankS,allow_pickle=True).tolist())
# ranks = [a.cpu().numpy() for a in rankS]

result = np.array(rankS)
# pdb.set_trace()
result = -np.sort(-result,1)
y = np.mean(result,0)
x = list(range(len(y)))
plt.plot(x,y,'k^-',label='MT+rankS+ISW+EW',color="red",linewidth=2) 
# std = np.std(result,0)
# r1 = list(map(lambda x: x[0]-x[1], zip(y, std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(y, std)))
# plt.fill_between(x, r1, r2, color="red", alpha=0.2)
#显示图示  
plt.legend()  
  
#显示图  
# plt.show()  
  
#保存图  
plt.savefig("aw.jpg")  
print('save')