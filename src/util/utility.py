import matplotlib.pyplot as plt
import numpy as np
import argparse
import pdb
def get_args():
    parser = argparse.ArgumentParser(description='Semi-POINT')
    # Training settings
    parser.add_argument('--p', type=str, default='',
                        help='input path')
    args = parser.parse_args()
    return args

def randomDrawer(path):
    data = np.load(path).tolist()
    keys = list(data.keys())
    pl = []
    for i in range(len(keys)):
        pl.append([i,data[keys[i]][0].item()])
    support = np.array(pl)    
    support_x = support[:,0]
    support_y = support[:,1]
    plt.plot(support_x, support_y, '-',label='w')
    plt.title('w vs. img counter')
    plt.ylabel('w')
    plt.legend(loc='best')
    plt.show()
    # pdb.set_trace()
if __name__ == '__main__':
    args = get_args()
    randomDrawer(args.p)

    