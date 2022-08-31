import os
import random
import json
from datetime import datetime as dt
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as kb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv3DTranspose, MaxPooling2D,MaxPooling3D, \
    concatenate, Input, Reshape, BatchNormalization, ELU, ReLU, Conv3D, LeakyReLU, Softmax

import utils.binvox_rw
import utils.datasetloader
import utils.preprocessing
import model.encoder
import model.decoder
import model.refiner
import model.loss
import model.merger
from utils.binvox_visualization import get_volume_views
import utils.data_transforms
def predict_net(cfg):
    import model.encoder
    import model.decoder
    import model.refiner
    import model.loss
    import model.merger
    encoder=model.encoder.gen_encoder_nonorm(cfg)
    #encoder.summary()
    decoder=model.decoder.gen_decoder_nonorm(cfg)
    #decoder.summary()
    print("establish networks")
    lenencoderdecoder=len(encoder.layers)+len(decoder.layers)+1
    Pix2VoxED = model.edcombiner.add_combiner(cfg,encoder,decoder)
    Pix2VoxR = model.refiner.add_refiner_nonorm(cfg, encoder, decoder)
    Pix2VoxM=model.merger.add_mergerv2(cfg, Pix2VoxR,lenencoderdecoder)
    
    
    '''
    test_image_path='./essay/bench/12.png'
    transforms = utils.data_transforms.Compose([
        #utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        #utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        #utils.data_transforms.ToTensor(),
    ])
    rendering_image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    rendering_image = cv2.resize(rendering_image, (cfg.CONST.IMG_H, cfg.CONST.IMG_W))
    rendering_image = np.expand_dims(rendering_image, axis=0)

    inp=transforms(rendering_image)
    Pix2Vox=Pix2VoxR
    Pix2Vox.load_weights('./rbest6550.h5')
    res = Pix2Vox.predict(inp)
    resR = np.squeeze(res[1],axis=0)
    resR = np.squeeze(resR,axis=3)
    resR = resR.transpose((0,2,1))
    resR = np.where(resR > 0.35, True, False)
    get_volume_views(resR, './essay/bench/', 0)
    with open('./essay/bench/model.binvox','rb') as f:
        model = utils.binvox_rw.read_as_3d_array(f)
    model.data=model.data.transpose((0,2,1))
    with open('./essay/bench/modelt.binvox','wb') as f:
        model.write(f)
    model.data=resR
    with open('./essay/bench/predict01.binvox','wb') as f:
        model.write(f)
    
    '''
    
    
    path='./essay/bench/'
    test_image_paths=[]
    test_image_paths.append(path+'00.png')
    test_image_paths.append(path+'01.png')
    test_image_paths.append(path+'02.png')
    test_image_paths.append(path+'03.png')
    test_image_paths.append(path+'04.png')
    test_image_paths.append(path+'05.png')
    test_image_paths.append(path+'06.png')
    test_image_paths.append(path+'07.png')
    test_image_paths.append(path+'08.png')
    test_image_paths.append(path+'09.png')
    test_image_paths.append(path+'10.png')
    test_image_paths.append(path+'11.png')
    test_image_paths.append(path+'12.png')
    test_image_paths.append(path+'13.png')
    test_image_paths.append(path+'14.png')
    test_image_paths.append(path+'15.png')
    test_image_paths.append(path+'16.png')
    test_image_paths.append(path+'17.png')
    test_image_paths.append(path+'18.png')
    test_image_paths.append(path+'19.png')
    transforms = utils.data_transforms.Compose([
        #utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        #utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        #utils.data_transforms.ToTensor(),
    ])
    for i in range(20):
        test_image_path=test_image_paths[i]
        rendering_image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        rendering_image = cv2.resize(rendering_image, (cfg.CONST.IMG_H, cfg.CONST.IMG_W))
            #rendering_images.append(rendering_image)
        rendering_image = np.expand_dims(rendering_image, axis=0)
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        if i==0:
            inputimg=transforms(rendering_image)
            inp1=inputimg
        else:
            inputimg=np.concatenate([inputimg,transforms(rendering_image)])
    inp8=np.expand_dims(inputimg, axis=0)
    print(inp1.shape)
    print(inp8.shape)
    
    '''
    for i in range(6):
        if i==0:
            Pix2Vox=Pix2VoxED
            Pix2Vox.load_weights('./log_onlyed/weights-impovement-epoch10--val_IoU0.5500.hdf5')
        elif i==1:
            Pix2Vox=Pix2VoxED
            Pix2Vox.load_weights('./log_onlyed/weights-impovement-epoch50--val_IoU0.6044.hdf5')
        elif i==2:
            Pix2Vox=Pix2VoxED
            Pix2Vox.load_weights('./log_onlyed/weights-impovement-epoch100--val_IoU0.6147.hdf5')
        elif i==3:
            Pix2Vox=Pix2VoxED
            Pix2Vox.load_weights('./log_onlyed/weights-impovement-epoch180--val_IoU0.6338.hdf5')
        elif i==4:
            Pix2Vox=Pix2VoxR
            Pix2Vox.load_weights('./log_r_and_ed/weights-impovement-epoch01--val_IoU0.6542.hdf5')
        elif i==5:
            Pix2Vox=Pix2VoxR
            Pix2Vox.load_weights('./rbest6550.h5')
        res = Pix2Vox.predict(inp)
        print(i)
        #input()
        if i==4 or i==5:
            resR = np.squeeze(res[1],axis=0)
        else:
            resR = np.squeeze(res,axis=0)
        resR = np.squeeze(resR,axis=3)
        resR = resR.transpose((0,2,1))
        resR = np.where(resR > 0.35, 1, 0)
        get_volume_views(resR, "./", i)
    '''
    '''
    inp=inp1
    Pix2Vox=Pix2VoxR
    Pix2Vox.load_weights('./rbest6550.h5')
    res = Pix2Vox.predict(inp)
    resR = np.squeeze(res[1],axis=0)
    resR = np.squeeze(resR,axis=3)
    resR = resR.transpose((0,2,1))
    resR = np.where(resR > 0.35, True, False)
    get_volume_views(resR, path, 0)
    with open(path+'model.binvox','rb') as f:
        model = utils.binvox_rw.read_as_3d_array(f)
    model.data=model.data.transpose((0,2,1))
    with open(path+'modelt.binvox','wb') as f:
        model.write(f)
    model.data=resR
    with open(path+'predict01.binvox','wb') as f:
        model.write(f)
    '''
    
    inp=inp8
    Pix2Vox=Pix2VoxM
    Pix2Vox.load_weights('./mbest.h5')
    res = Pix2Vox.predict(inp)
    resR = np.squeeze(res[1],axis=0)
    resR = np.squeeze(resR,axis=3)
    resR = resR.transpose((0,2,1))
    resR = np.where(resR > 0.35, True, False)
    get_volume_views(resR, path, 1)
    with open(path+'model.binvox','rb') as f:
        model = utils.binvox_rw.read_as_3d_array(f)
    model.data=resR
    with open(path+'predictn20.binvox','wb') as f:
        model.write(f)

    





