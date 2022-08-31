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
import tensorflowjs as tfjs
def convert_net(cfg):
    encoder=model.encoder.gen_encoder_nonorm(cfg)
    #encoder.summary()
    decoder=model.decoder.gen_decoder_nonorm(cfg)
    #decoder.summary()
    print("establish networks")
    lenencoderdecoder=len(encoder.layers)+len(decoder.layers)+1
    Pix2VoxED = model.edcombiner.add_combiner(cfg,encoder,decoder)
    Pix2VoxED.load_weights('./log_onlyed/weights-impovement-epoch250--val_IoU0.6360.hdf5')
    Pix2VoxR = model.refiner.add_refiner_nonorm(cfg, encoder, decoder)
    del(encoder)
    del(decoder)
    Pix2Vox=Pix2VoxED
    #Pix2Vox.load_weights('./mybest.h5')
    #Pix2Vox1.summary()
    #input()

    #Pix2Vox = model.merger.add_merger(cfg,Pix2Vox1)
    #del(Pix2Vox1)
    Pix2Vox.summary(line_length=200)
    
    tfjs.converters.save_keras_model(Pix2Vox, "jsmodel_small")
    #Pix2Voxsmall.save("p2vsmall", save_format='tf')
    #input()
    #2.0

