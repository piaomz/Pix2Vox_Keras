import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv3DTranspose, MaxPooling2D,MaxPooling3D, \
    concatenate, Input, Reshape, BatchNormalization, ELU, ReLU, Conv3D, LeakyReLU,Lambda, Dropout

from tensorflow.keras import initializers
zeroinit=initializers.Zeros()
oneinit=initializers.Ones()
heinit=initializers.he_normal()
normalinit=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)


def add_refiner_m(cfg,modelm,modelr,lenencoderdecoder):
    inputm=Input(shape=(cfg.CONST.N_VOX, cfg.CONST.N_VOX, cfg.CONST.N_VOX,1))
    x = Conv3D(32, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(inputm)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x16 = MaxPooling3D((2, 2, 2))(x)
    x = x16
    x = Conv3D(64, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x8 = MaxPooling3D((2, 2, 2))(x)
    x = x8
    x = Conv3D(128, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x4 = MaxPooling3D((2, 2, 2))(x)
    x = x4
    x = Flatten()(x)
    x = Dense(2048,kernel_initializer=normalinit,bias_initializer=zeroinit)(x)
    x = ReLU()(x)
    x = Dense(8192,kernel_initializer=normalinit,bias_initializer=zeroinit)(x)
    x = ReLU()(x)
    x = Reshape((4, 4, 4, 128))(x)
    x= Conv3DTranspose(64, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x+x4)
    x = ReLU()(x)
    x = Conv3DTranspose(32, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x + x8)
    x = ReLU()(x)
    x = Conv3DTranspose(1, (4, 4, 4), padding='SAME', strides=(2, 2, 2),activation='sigmoid',kernel_initializer=heinit,bias_initializer=zeroinit)(x + x16)
    x = (x+inputm)*0.5
    x = Lambda(lambda x: x, name='R')(x)
    refiner=Model(inputm,x)
    for i in range(1,len(refiner.layers)):
        refiner.layers[i].set_weights(modelr.layers[lenencoderdecoder+i].get_weights())
    fullinput = Input(shape=(cfg.CONST.N_VIEWS_RENDERING,cfg.CONST.IMG_H, cfg.CONST.IMG_W, 3))
    mres=modelm(fullinput)
    output=refiner(mres)
    Pix2Vox= Model(fullinput,[mres,output])
    return Pix2Vox



def add_refiner_nonorm(cfg,encoder,decoder):
  # connect encoder decoder and refiner
    input_x = Input(shape=(cfg.CONST.IMG_W, cfg.CONST.IMG_H, 3))
    x = input_x
    for i in range(len(encoder.layers)):
        x = encoder.layers[i](x)
    x = Reshape((2, 2, 2, 2048))(x)
    for j in range(len(decoder.layers)):
        x = decoder.layers[j](x)
    x32 = x
    #refiner_input = Input(shape=(32, 32, 32, 1))
    #x = refiner_input
    x = Conv3D(32, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x16 = MaxPooling3D((2, 2, 2))(x)
    #x = Dropout(0.2)(x)
    x = x16
    x = Conv3D(64, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x8 = MaxPooling3D((2, 2, 2))(x)
    #x = Dropout(0.2)(x)
    x = x8
    x = Conv3D(128, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x4 = MaxPooling3D((2, 2, 2))(x)
    #x = Dropout(0.2)(x)
    x = x4
    x = Flatten()(x)
    x = Dense(2048,kernel_initializer=normalinit,bias_initializer=zeroinit)(x)
    x = ReLU()(x)
    #x = Dropout(0.2)(x)
    x = Dense(8192,kernel_initializer=normalinit,bias_initializer=zeroinit)(x)
    x = ReLU()(x)
    #x = Dropout(0.2)(x)
    x = Reshape((4, 4, 4, 128))(x)
    x= Conv3DTranspose(64, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x+x4)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = Dropout(0.1)(x)
    x = Conv3DTranspose(32, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x + x8)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = Dropout(0.1)(x)
    x = Conv3DTranspose(1, (4, 4, 4), padding='SAME', strides=(2, 2, 2),activation='sigmoid',kernel_initializer=heinit,bias_initializer=zeroinit)(x + x16)
    x = (x+x32)*0.5
    x = Lambda(lambda x: x, name='R')(x)
    auto_encoder = Model(input_x, [x32,x])
    #auto_encoder = Model(input_x, x)
    return auto_encoder


def add_refinerv2(cfg,edcoder):
  # connect encoder decoder and refiner
    input_x = Input(shape=(cfg.CONST.IMG_W, cfg.CONST.IMG_H, 3))
    x = input_x
    for i in range(len(edcoder.layers)):
        edcoder.layers[i].trainable = False
        x = encoder.layers[i](x)
    x32 = x
    #refiner_input = Input(shape=(32, 32, 32, 1))
    #x = refiner_input
    x = Conv3D(32, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x16 = MaxPooling3D((2, 2, 2))(x)
    x = x16
    x = Conv3D(64, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x8 = MaxPooling3D((2, 2, 2))(x)
    x = x8
    x = Conv3D(128, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x4 = MaxPooling3D((2, 2, 2))(x)
    x = x4
    x = Flatten()(x)
    x = Dense(2048,kernel_initializer=normalinit,bias_initializer=zeroinit)(x)
    x = ReLU()(x)
    x = Dense(8192,kernel_initializer=normalinit,bias_initializer=zeroinit)(x)
    x = ReLU()(x)
    x = Reshape((4, 4, 4, 128))(x)
    x= Conv3DTranspose(64, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x+x4)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(32, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x + x8)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(1, (4, 4, 4), padding='SAME', strides=(2, 2, 2),activation='sigmoid',kernel_initializer=heinit,bias_initializer=zeroinit)(x + x16)
    x = (x+x32)*0.5
    x = Lambda(lambda x: x, name='R')(x)
    auto_encoder = Model(input_x, x)
    return auto_encoder

def add_refiner(cfg,encoder,decoder):
    # connect encoder decoder and refiner
    input_x = Input(shape=(cfg.CONST.IMG_W, cfg.CONST.IMG_H, 3))
    x = input_x
    for i in range(len(encoder.layers)):
        x = encoder.layers[i](x)
    x = Reshape((2, 2, 2, 2048))(x)
    for j in range(len(decoder.layers)):
        x = decoder.layers[j](x)
    x32 = x
    #refiner_input = Input(shape=(32, 32, 32, 1))
    #x = refiner_input
    x = Conv3D(32, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x16 = MaxPooling3D((2, 2, 2))(x)
    x = x16
    x = Conv3D(64, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x8 = MaxPooling3D((2, 2, 2))(x)
    x = x8
    x = Conv3D(128, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x4 = MaxPooling3D((2, 2, 2))(x)
    x = x4
    x = Flatten()(x)
    x = Dense(2048,kernel_initializer=normalinit,bias_initializer=zeroinit)(x)
    x = ReLU()(x)
    x = Dense(8192,kernel_initializer=normalinit,bias_initializer=zeroinit)(x)
    x = ReLU()(x)
    x = Reshape((4, 4, 4, 128))(x)
    x= Conv3DTranspose(64, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x+x4)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(32, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x + x8)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(1, (4, 4, 4), padding='SAME', strides=(2, 2, 2),activation='sigmoid',kernel_initializer=heinit,bias_initializer=zeroinit)(x + x16)
    x = (x+x32)*0.5
    x = Lambda(lambda x: x, name='R')(x)
    auto_encoder = Model(input_x, [x32,x])
    return auto_encoder