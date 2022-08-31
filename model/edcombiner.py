import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv3DTranspose, MaxPooling2D,MaxPooling3D, \
    concatenate, Input, Reshape, BatchNormalization, ELU, ReLU, Conv3D, LeakyReLU,Lambda

def add_combiner(cfg,encoder,decoder):
    # connect encoder decoder and refiner
    input_x = Input(shape=(cfg.CONST.IMG_W, cfg.CONST.IMG_H, 3))
    x = input_x
    for i in range(len(encoder.layers)):
        x = encoder.layers[i](x)
    x = Reshape((2, 2, 2, 2048))(x)
    for j in range(len(decoder.layers)):
        x = decoder.layers[j](x)
    auto_encoder = Model(input_x,x)
    return auto_encoder
    
def add_combiner_f(cfg,encoder,decoder):
    # connect encoder decoder and refiner
    input_x = Input(shape=(cfg.CONST.IMG_W, cfg.CONST.IMG_H, 3))
    x = input_x
    for i in range(len(encoder.layers)):
        x = encoder.layers[i](x)
    x = Reshape((2, 2, 2, 256))(x)
    for j in range(len(decoder.layers)):
        x = decoder.layers[j](x)
    auto_encoder = Model(input_x,x)
    return auto_encoder