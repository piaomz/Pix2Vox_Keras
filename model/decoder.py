import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv3DTranspose, MaxPooling2D, concatenate, Input, \
    Reshape, BatchNormalization, ELU, ReLU, Dropout
from tensorflow.keras import initializers
zeroinit=initializers.Zeros()
oneinit=initializers.Ones()
heinit=initializers.he_normal()
def gen_decoder(cfg):
    decoder = Sequential()
    decoder.add(Input(shape=(2, 2, 2, 2048)))
    decoder.add(Conv3DTranspose(512, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    decoder.add(BatchNormalization())
    decoder.add(ReLU())
    decoder.add(Conv3DTranspose(128, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    decoder.add(BatchNormalization())
    decoder.add(ReLU())
    decoder.add(Conv3DTranspose(32, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    decoder.add(BatchNormalization())
    decoder.add(ReLU())
    decoder.add(Conv3DTranspose(8, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    decoder.add(BatchNormalization())
    decoder.add(ReLU())
    decoder.add(Conv3DTranspose(1, (1, 1, 1), padding='SAME', strides=(1, 1, 1), activation='sigmoid', name='ED',kernel_initializer=heinit,bias_initializer=zeroinit))
    return decoder

def gen_decoder_nonorm(cfg):
    decoder = Sequential()
    decoder.add(Input(shape=(2, 2, 2, 2048)))
    decoder.add(Conv3DTranspose(512, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    #decoder.add(BatchNormalization())
    decoder.add(ReLU())
    #decoder.add(Dropout(0.1))
    decoder.add(Conv3DTranspose(128, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    #decoder.add(BatchNormalization())
    decoder.add(ReLU())
    #decoder.add(Dropout(0.05))
    decoder.add(Conv3DTranspose(32, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    #decoder.add(BatchNormalization())
    decoder.add(ReLU())
    #decoder.add(Dropout(0.05))
    decoder.add(Conv3DTranspose(8, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    #decoder.add(BatchNormalization())
    decoder.add(ReLU())
    #decoder.add(Dropout(0.05))
    decoder.add(Conv3DTranspose(1, (1, 1, 1), padding='SAME', strides=(1, 1, 1), activation='sigmoid', name='ED',kernel_initializer=heinit,bias_initializer=zeroinit))
    return decoder


def gen_decoder_nonorm_f(cfg):
    decoder = Sequential()
    decoder.add(Input(shape=(2, 2, 2, 256)))
    decoder.add(Conv3DTranspose(128, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    #decoder.add(BatchNormalization())
    decoder.add(ReLU())
    #decoder.add(Dropout(0.1))
    decoder.add(Conv3DTranspose(64, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    #decoder.add(BatchNormalization())
    decoder.add(ReLU())
    #decoder.add(Dropout(0.1))
    decoder.add(Conv3DTranspose(32, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    #decoder.add(BatchNormalization())
    decoder.add(ReLU())
    #decoder.add(Dropout(0.1))
    decoder.add(Conv3DTranspose(8, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit))
    #decoder.add(BatchNormalization())
    decoder.add(ReLU())
    #decoder.add(Dropout(0.1))
    decoder.add(Conv3DTranspose(1, (1, 1, 1), padding='SAME', strides=(1, 1, 1), activation='sigmoid', name='ED',kernel_initializer=heinit,bias_initializer=zeroinit))
    return decoder


