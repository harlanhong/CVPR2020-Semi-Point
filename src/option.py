import argparse

parser = argparse.ArgumentParser(description='Semi-POINT')
# Training settings
parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--nw', type=int, default=4,
                    help='number of workers')
parser.add_argument('--ni', type=int, default=8,
                    help='number of instances for each tracklet (default is 8)')
parser.add_argument('--h',type=int, default=4,help='the number of relation submodule')
parser.add_argument('--N',type=int,default=2,help='the number of relation module')
parser.add_argument('--save_name',type=str,default='POINT',help='the name of saved model')
# parser.add_argument('--index_name',type=str,default='/home/share/fating/ProcessDataset/MSindex.npy',help='the name of index of train, val and test set')
# parser.add_argument('--dataset_path',type=str,default='/home/share/fating/ProcessDataset/MSDataSet_process',help='the path of dataset')

#for pre-train
parser.add_argument('--pre-train',action='store_true',help='adopt the pre-train way')
#for NCAA
# parser.add_argument('--dataset_path',type=str,default='/data/fating/ProcessDataset/NCAAv2_process',help='the path of dataset')
# parser.add_argument('--index_name',type=str,default='/data/fating/OriginalDataset/NCAAv2/data/ENCAA.npy',help='the name of index of train, val and test set')
#for MS
parser.add_argument('--dataset_path',type=str,default='/data/fating/ProcessDataset/MSDatasetV2_process',help='the path of dataset')
parser.add_argument('--index_name',type=str,default='/data/fating/OriginalDataset/MSDatasetv2/data/MSexpand_DSFD.npy',help='the name of index of train, val and test set')


#for pre-trained



parser.add_argument('--model',type=str,default='',help='model name')

#for data
parser.add_argument('--sample_size',type=int,default=1,help='the size of image selected from dataset')
parser.add_argument('--width',type=int,default=224,help='the width of image')
parser.add_argument('--height',type=int,default=224,help='the height of image')


#for optimizer
parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
parser.add_argument('--optimizer', default='SGD', choices=('SGD','ADAM','NADAM','RMSprop'), help='optimizer to use (SGD | ADAM | NADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--dampening', type=float, default=0, help='SGD dampening')
parser.add_argument('--nesterov', action='store_true', help='SGD nesterov')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--amsgrad', action='store_true', help='ADAM amsgrad')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')

#for lp
parser.add_argument('--sigma', type=float, default=0.5)


#mixMatch
parser.add_argument('--K',type=int,default=3,help='number of unlabeled augmentations')
parser.add_argument('--T',type=float,default=0.5,help='the sharpening temperature')
parser.add_argument('--prob_threshold',type=float,default=0.7,help='the threshold of the high probability item')

#mean-teacher
parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
parser.add_argument('--consistency', default=1, type=float, metavar='WEIGHT',
                    help='use consistency loss with given weight (default: None)')
parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                    choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--consistency-rampup', type=int, metavar='EPOCHS',
                    help='length of the consistency loss ramp-up')

parser.add_argument('--rampup-type', type=str, default='sigmoid_rampup',
                    help='length of the consistency loss ramp-up')

parser.add_argument('--consistency-loss', default='Learn_mse_loss', type=str, 
                        help='the consistency function')
parser.add_argument('--label-propagation', default='LabelPropagation', type=str, 
                        help='the label propagation function')
parser.add_argument('--rampup-per-step',action='store_true',help='ramp up the consistency weight per training step')


parser.add_argument('--interval',type=int,default=2)

parser.add_argument('--training-step',type=int,default=1000)
parser.add_argument('--vip-num',type=int,default=1)
parser.add_argument('--AW-thresh',type=float,default=0.05)
parser.add_argument('--third-loss',type=str,default=0.05)
parser.add_argument('--threshold',type=float,default=0.99)
parser.add_argument('--test-savename',type=str)
parser.add_argument('--unlabelled',action='store_true')
parser.add_argument('--testID',type=str)

parser.add_argument('--origin-uw',action='store_true',help='the type of uw')

parser.add_argument('--pretrain_model',type=str)


args = parser.parse_args()
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

