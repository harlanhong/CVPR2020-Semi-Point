from collections import defaultdict
import numpy as np
import pdb
import matplotlib.pyplot as plt
class Recorder():
    def __init__(self,name):
        self.record = defaultdict(list)
        self.name = name
    def update(self,key,data,epoch):
        self.record[key].append([epoch,data])
    def save(self):
        np.save('../exp/'+self.name+'.pkl',self.record)
class Drawer():
    def __init__(self,path,type='-',label='none'):
        self.record = np.load(path).tolist()
    def plot(self,key,type,label):
        support = np.array(self.record[key])
        support_x = support[:,0]
        support_y = support[:,1]
        if 'iteration' in key or 'lambda' in key:
            support_y = self.movingaverage(support_y, 500)
        # pdb.set_trace()
        # xnew = np.linspace(support_x.min(),support_x.max(),600)
        # power_smooth = spline(support_x,support_y,xnew)
        # plt.plot(xnew, power_smooth, type,label=label)
        plt.plot(support_x, support_y, type,label=label)
    def movingaverage(self,interval, window_size):
        window= np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
    def draw(self):
        keys = list(self.record.keys())
        # plot loss
        loss_keys = [key for key in keys if 'loss' in key]
        
        for key in loss_keys:
            # if 'val' in key:
            #     continue
            # if 'test' in key:
            #     continue
            # if 'iteration labelled loss' in key:
            #     continue
            # if 'iteration unlabelled loss' in key:
            #     continue
            # if 'iteration lambda unlabelled loss' in key:
            #     continue
            self.plot(key,'-',key)
            plt.title('Ours loss vs. epoches')
            plt.ylabel('loss')
            plt.legend(loc='best')
            print(key)
            plt.savefig('./img/'+key+'.pdf')
            plt.figure()
        totalloss = np.array(self.record['iteration lambda unlabelled loss'])[:,1] +np.array(self.record['iteration labelled loss'])[:,1]
        x = range(len(totalloss))
        y = self.movingaverage(totalloss, 500)
        plt.plot(x, y, '-',label='total loss')
        plt.title('Ours loss vs. epoches')
        plt.ylabel('loss')
        plt.legend(loc='best')
        print('total loss')
        plt.savefig('./img/total-loss'+'.pdf')
        plt.figure()
        #plot acc
        acc_keys = [key for key in keys if 'acc' in key]
        for key in acc_keys:
           
            self.plot(key,'-',key)
            plt.title('acc vs. epoches')
            plt.ylabel('acc')
            plt.legend(loc='best')
            print(key)

            plt.savefig('./img/'+key+'.pdf')
            plt.figure()
        #plot mAP:
        mAP_keys = [key for key in keys if 'mAP' in key]
        for key in mAP_keys:
           
            self.plot(key,'-',key)
            plt.title('mAP vs. epoches')
            plt.ylabel('mAP')
            plt.legend(loc='best')
            print(key)

            plt.savefig('./img/'+key+'.pdf')
            plt.figure()
       
    def draw2(self):
        support = np.array(self.record['iteration unlabelled loss'])
        support_x = support[:,0]
        support_y = support[:,1]
        epoches = int(len(support_x)/4189)
        pdb.set_trace()

        support_x = support_x[:epoches*4189]

        support_x = support_x.reshape(epoches,4189)

        support_x = np.mean(support_x,0)
        # for i in range(int(len(support_x)/4189)):
    
    def draw3(self):
        support = np.array(self.record['unlabel loss'])
        support_x = support[:,0]
        support_y = support[:,1]
        epoches = int(len(support_y)/38)
        support_y = support_y[:epoches*38]
        support_y = support_y.reshape(epoches,38)
        support_y = np.mean(support_y,1)
        support_x = support_x[:len(support_y)]
        plt.plot(support_x, support_y, '-',label='unlabelled loss')
        plt.title('Ours loss vs. epoches')
        plt.ylabel('train loss')
        plt.legend(loc='best')
        plt.show()
    def draw3(self):
        support = np.array(self.record['unlabel loss'])
        support_x = support[:,0]
        support_y = support[:,1]
        epoches = int(len(support_y)/18)
        support_y = support_y[:epoches*18]
        support_y = support_y.reshape(epoches,18)
        support_y = np.mean(support_y,1)
        support_x = support_x[:len(support_y)]
        plt.plot(support_x, support_y, '-',label='unlabelled loss')
        plt.title('Ours loss vs. epoches')
        plt.ylabel('train loss')
        plt.legend(loc='best')
        plt.show()
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Semi-POINT')
    # Training settings
    parser.add_argument('-p', type=str,default='')
    args = parser.parse_args()

    drawer = Drawer(args.p)
    drawer.draw()
    


        