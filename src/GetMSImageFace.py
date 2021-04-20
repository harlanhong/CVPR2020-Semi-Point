import os
import numpy as np
import random

import skimage
from skimage import data, io
# import skimage.io
import skimage.transform
import skimage.color
import pdb
import pickle
import fnmatch
from PIL import Image
import scipy.io as scio
from tqdm import tqdm

#Image size after processing
SIZE_WIDTH = 224
SIZE_HEIGHT = 224
#the directory save you preprocess data
saveDir = '/home/share/fating/OriginalDataset/MSDatasetV2_process/'

if not os.path.exists(saveDir):
    os.makedirs(saveDir)
#The dataset
# matFile = '/home/share/fating/OriginalDataset/MSDataSet/data/annotations'
LableFile = '/home/share/fating/OriginalDataset/MSDatasetv2/data/MSexpand_DSFD.npy'
ImagePath = '/home/share/fating/OriginalDataset/MSDatasetv2/images/'
data = np.load(LableFile).tolist()
train = data['train']
val = data['val']
test = data['test']
keys_train = list(train.keys())
keys_test = list(test.keys())
keys_val = list(val.keys())

num_train = len(keys_train)
num_val = len(keys_val)
num_test = len(keys_test)


#process train set
for idx , key in enumerate(tqdm(keys_train)):
    name = key
    bboxes = train[key]
    FaceFolderName = saveDir + 'Image_'+name+ '/Face'
    if not os.path.exists(FaceFolderName):
        os.makedirs(FaceFolderName)
    FaceContFolderName = saveDir + 'Image_'+name + '/FaceCont'
    if not os.path.exists(FaceContFolderName):
        os.makedirs(FaceContFolderName)
    CoorFolderName = saveDir + 'Image_'+name + '/Coordinate'
    if not os.path.exists(CoorFolderName):
        os.makedirs(CoorFolderName)
    FullImgFolderName = saveDir + 'Image_'+name + '/Image'
    if not os.path.exists(FullImgFolderName):
        os.makedirs(FullImgFolderName)
    NumFace = len(bboxes)
    img = Image.open(ImagePath+name+'.jpg').convert('RGB')
    width,height = img.size

    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = bboxes[j]
        xMin, yMin, xMax, yMax,label = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3]),int(Rect[4])
        w = xMax-xMin
        h = yMax-yMin
        c_xMin = int(max(1,int(xMin-2.5*w)))
        c_xMax = int(min(width, int(xMax+2.5*w)))
        c_yMin = max(1,yMin-1.5*h)
        c_yMax = min(height, yMax+5.5*h)
        TempFace = img.crop([xMin,yMin,xMax,yMax]).resize([SIZE_WIDTH,SIZE_HEIGHT])
        No_Face = '0' + str(j)
        FaceName = FaceFolderName+'/' + 'Image_' + name + \
                   '_Face_' + No_Face[len(No_Face)-2:] + '_Label_' + str(int(label)) + '.jpg'
        TempFace.save(FaceName)
        TempFaceCont = img.crop([c_xMin,c_yMin,c_xMax,c_yMax]).resize([SIZE_WIDTH,SIZE_HEIGHT])
        # TempFaceCont = img.crop([c_yMin,c_xMin,c_yMax,c_xMax]).resize([224,224])
        FaceContName = FaceContFolderName+'/' + 'Image_' \
                       + name + '_Face_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempFaceCont.save(FaceContName)
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[yMin:yMax, xMin:xMax] = 255
        TempCoor = Image.fromarray(np.uint8(canvas))
        TempCoor = TempCoor.resize([SIZE_WIDTH,SIZE_HEIGHT])
        CoorName = CoorFolderName+'/' + 'Image_' \
                       + name + '_Coor_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempCoor.save(CoorName)
        ImgName = FullImgFolderName+'/' + 'Image_' \
                       + name + '_Img_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        imgCopy.save(ImgName)
#process val set
for idx , key in enumerate(tqdm(keys_val)):
    name = key
    bboxes = val[key]
    FaceFolderName = saveDir + 'Image_'+name+ '/Face'
    if not os.path.exists(FaceFolderName):
        os.makedirs(FaceFolderName)
    FaceContFolderName = saveDir + 'Image_'+name + '/FaceCont'
    if not os.path.exists(FaceContFolderName):
        os.makedirs(FaceContFolderName)
    CoorFolderName = saveDir + 'Image_'+name + '/Coordinate'
    if not os.path.exists(CoorFolderName):
        os.makedirs(CoorFolderName)
    FullImgFolderName = saveDir + 'Image_'+name + '/Image'
    if not os.path.exists(FullImgFolderName):
        os.makedirs(FullImgFolderName)
    NumFace = len(bboxes)
    img = Image.open(ImagePath+name+'.jpg').convert('RGB')
    width,height = img.size
    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = bboxes[j]
        xMin, yMin, xMax, yMax,label = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3]),int(Rect[4])
        w = xMax-xMin
        h = yMax-yMin
        c_xMin = int(max(1,int(xMin-2.5*w)))
        c_xMax = int(min(width, int(xMax+2.5*w)))
        c_yMin = max(1,yMin-1.5*h)
        c_yMax = min(height, yMax+5.5*h)
        TempFace = img.crop([xMin,yMin,xMax,yMax]).resize([SIZE_WIDTH,SIZE_HEIGHT])
        No_Face = '0' + str(j)
        FaceName = FaceFolderName+'/' + 'Image_' + name + \
                   '_Face_' + No_Face[len(No_Face)-2:] + '_Label_' + str(int(label)) + '.jpg'
        TempFace.save(FaceName)
        TempFaceCont = img.crop([c_xMin,c_yMin,c_xMax,c_yMax]).resize([SIZE_WIDTH,SIZE_HEIGHT])
        # TempFaceCont = img.crop([c_yMin,c_xMin,c_yMax,c_xMax]).resize([224,224])
        FaceContName = FaceContFolderName+'/' + 'Image_' \
                       + name + '_Face_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempFaceCont.save(FaceContName)
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[yMin:yMax, xMin:xMax] = 255
        TempCoor = Image.fromarray(np.uint8(canvas))
        TempCoor = TempCoor.resize([SIZE_WIDTH,SIZE_HEIGHT])
        CoorName = CoorFolderName+'/' + 'Image_' \
                       + name + '_Coor_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempCoor.save(CoorName)
        ImgName = FullImgFolderName+'/' + 'Image_' \
                       + name + '_Img_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        imgCopy.save(ImgName)
#process test set
for idx , key in enumerate(tqdm(keys_test)):
    name = key
    bboxes = test[key]
    FaceFolderName = saveDir + 'Image_'+name+ '/Face'
    if not os.path.exists(FaceFolderName):
        os.makedirs(FaceFolderName)
    FaceContFolderName = saveDir + 'Image_'+name + '/FaceCont'
    if not os.path.exists(FaceContFolderName):
        os.makedirs(FaceContFolderName)
    CoorFolderName = saveDir + 'Image_'+name + '/Coordinate'
    if not os.path.exists(CoorFolderName):
        os.makedirs(CoorFolderName)
    FullImgFolderName = saveDir + 'Image_'+name + '/Image'
    if not os.path.exists(FullImgFolderName):
        os.makedirs(FullImgFolderName)
    NumFace = len(bboxes)
    img = Image.open(ImagePath+name+'.jpg').convert('RGB')
    width,height = img.size

    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = bboxes[j]
        xMin, yMin, xMax, yMax,label = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3]),int(Rect[4])
        w = xMax-xMin
        h = yMax-yMin
        c_xMin = int(max(1,int(xMin-2.5*w)))
        c_xMax = int(min(width, int(xMax+2.5*w)))
        c_yMin = max(1,yMin-1.5*h)
        c_yMax = min(height, yMax+5.5*h)
        TempFace = img.crop([xMin,yMin,xMax,yMax]).resize([SIZE_WIDTH,SIZE_HEIGHT])
        No_Face = '0' + str(j)
        FaceName = FaceFolderName+'/' + 'Image_' + name + \
                   '_Face_' + No_Face[len(No_Face)-2:] + '_Label_' + str(int(label)) + '.jpg'
        TempFace.save(FaceName)
        TempFaceCont = img.crop([c_xMin,c_yMin,c_xMax,c_yMax]).resize([SIZE_WIDTH,SIZE_HEIGHT])
        # TempFaceCont = img.crop([c_yMin,c_xMin,c_yMax,c_xMax]).resize([224,224])
        FaceContName = FaceContFolderName+'/' + 'Image_' \
                       + name + '_Face_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempFaceCont.save(FaceContName)
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[yMin:yMax, xMin:xMax] = 255
        TempCoor = Image.fromarray(np.uint8(canvas))
        TempCoor = TempCoor.resize([SIZE_WIDTH,SIZE_HEIGHT])
        CoorName = CoorFolderName+'/' + 'Image_' \
                       + name + '_Coor_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        TempCoor.save(CoorName)
        ImgName = FullImgFolderName+'/' + 'Image_' \
                       + name + '_Img_' + No_Face[len(No_Face) - 2:] + \
                       '_Label_' + str(int(label)) + '.jpg'
        imgCopy.save(ImgName)
np.save('MSindex.npy', {'train': trainSet, 'val': valSet, 'test': testSet})
