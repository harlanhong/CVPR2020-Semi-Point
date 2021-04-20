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
saveDir = '/home/share/fating/OriginalDataset/NCAAv2_process/'

if not os.path.exists(saveDir):
    os.makedirs(saveDir)
#The dataset
# matFile = '/home/share/fating/OriginalDataset/MSDataSet/data/annotations'
LableFile = '/home/share/fating/OriginalDataset/NCAAv2/data/ENCAA.npy'
ImagePath = '/home/share/fating/OriginalDataset/NCAAv2/images/'
data = np.load(LableFile).tolist()
unlabel_train = data['unlabel_data']
label_train = data['label_data']
val = data['val']
test = data['test']
keys_unlabel_train = list(unlabel_train.keys())
keys_label_train = list(label_train.keys())
keys_test = list(test.keys())
keys_val = list(val.keys())

num_unlabel_train = len(keys_unlabel_train)
num_label_train = len(keys_label_train)
num_val = len(keys_val)
num_test = len(keys_test)


#process unlabel train set
for idx , key in enumerate(tqdm(keys_unlabel_train)):
    name = key
    print(key)
    bboxes = unlabel_train[key]
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
    img = Image.open(ImagePath+name).convert('RGB')
    width,height = img.size

    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = bboxes[j]
        xMin, yMin, xMax, yMax,label = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3]),int(Rect[4])
        label=0
        w = xMax-xMin
        h = yMax-yMin

        c_xMin = int(max(1,int(xMin-0.5*w)))
        c_xMax = int(min(width, int(xMin+1.5*w)))
        c_yMin = max(1,int(yMin-0.5*h))
        c_yMax = min(height, int(yMin+1.5*h))
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

#process unlabel train set
for idx , key in enumerate(tqdm(keys_label_train)):
    name = key
    print(key)

    bboxes = label_train[key]
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
    img = Image.open(ImagePath+name).convert('RGB')
    width,height = img.size

    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = bboxes[j]
        xMin, yMin, xMax, yMax,label = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3]),int(Rect[4])
        w = xMax-xMin
        h = yMax-yMin

        c_xMin = int(max(1,int(xMin-0.5*w)))
        c_xMax = int(min(width, int(xMin+1.5*w)))
        c_yMin = max(1,int(yMin-0.5*h))
        c_yMax = min(height, int(yMin+1.5*h))
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
    print(key)

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
    img = Image.open(ImagePath+name).convert('RGB')
    width,height = img.size
    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = bboxes[j]
        xMin, yMin, xMax, yMax,label = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3]),int(Rect[4])
        w = xMax-xMin
        h = yMax-yMin
        c_xMin = int(max(1,int(xMin-0.5*w)))
        c_xMax = int(min(width, int(xMin+1.5*w)))
        c_yMin = max(1,int(yMin-0.5*h))
        c_yMax = min(height, int(yMin+1.5*h))
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
    print(key)

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
    img = Image.open(ImagePath+name).convert('RGB')
    width,height = img.size

    imgCopy = img.resize([SIZE_WIDTH,SIZE_HEIGHT])
    for j in range(NumFace):
        Rect = bboxes[j]
        xMin, yMin, xMax, yMax,label = int(Rect[0]), int(Rect[1]), int(Rect[2]), int(Rect[3]),int(Rect[4])
        w = xMax-xMin
        h = yMax-yMin
        c_xMin = int(max(1,int(xMin-0.5*w)))
        c_xMax = int(min(width, int(xMin+1.5*w)))
        c_yMin = max(1,int(yMin-0.5*h))
        c_yMax = min(height, int(yMin+1.5*h))
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

