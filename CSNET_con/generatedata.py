'''
Generate the training data.
'''
import os
import random

import torch
import numpy as np
from PIL import Image

def data_augmentation(img, mode):
    if mode == 0:
        return img

    if mode == 1: # flipped
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    if mode == 2: # rotation 90
        img = img.transpose(Image.ROTATE_90)
        return img
    if mode == 3: # rotation 90 & flipped
        img = img.transpose(Image.ROTATE_90)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
    if mode == 4: # rotation 180
        img = img.transpose(Image.ROTATE_180)
        return img
    if mode == 5: # rotation 180 & flipped
        img = img.transpose(Image.ROTATE_180)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    if mode == 6: # rotation 270
        img = img.transpose(Image.ROTATE_270)
        return img

    if mode == 7: # rotation 270 & flipped
        img = img.transpose(Image.ROTATE_270)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img



def patches_generation(size_input, stride, folder, mode, max_numPatches, batchSize):
    count = 0
    # padding = abs(size_input - size_label) / 2
    if not os.path.exists('./patch_data_96'):
        os.mkdir('./patch_data_96')
    if mode==0:
        file_path_1 = './patch_data_96/train'
        if not os.path.exists(file_path_1):
            os.mkdir(file_path_1)
    else:
        file_path_1 = './patch_data_96/val'
        if not os.path.exists(file_path_1):
            os.mkdir(file_path_1)

    filepaths = []
    for file in os.listdir(folder):
        filepaths.append(os.path.join(folder,file))

    for path in filepaths:
        image = Image.open(path) # uint8
        # [~, name, exte] = fileparts(filepaths(i).name)
        #if image.mode == 'RGB':
            ## 转换灰度图像有两种方式，一种是RGB通道相同的值，另一种是从YCbCr空间中读取第一个通道
            #image = image.convert('YCbCr')
            #image = np.array(image)
            #image = image[:, :, 0]
            #image = Image.fromarray(image) # uint8

        for j in range(0,1):
            image_aug = data_augmentation(image, j) # augment
            [hei, wid] = image_aug.size
            im_input = np.array(image_aug)  # single
            for y in np.arange(0, hei-size_input, stride):
                for x in np.arange(0, wid-size_input, stride):
                    subim_input = im_input[x:x+size_input, y:y+size_input]
                    count = count + 1
                    img_path = os.path.join(file_path_1, str('%06d' % count) + '.jpg')
                    img_patches = Image.fromarray(subim_input)
                    img_patches.save(img_path)

    return 0


if __name__=="__main__":

    batchSize      = 60      ###batch size
    max_numPatches = batchSize*813
    modelName      = 'model_96_Adam'
    ## sigma          = 25  Gaussian noise level

    ### training and val
    folder_train  = 'F:/CSNET_LITE/data/train'  # training
    folder_val   = 'F:/CSNET_LITE/data/val'   # val
    size_input    = 96          # training
    stride_train  = 57          # training
    stride_test   = 60         # testing
    val_train     = 0           # training # default
    val_test      = 1           # testing  # default

    # training patches
    patches_generation(size_input,stride_train,folder_train,val_train,max_numPatches,batchSize)
    # testing  patches
    patches_generation(size_input,stride_test,folder_val,val_test,max_numPatches,batchSize)
