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
def gen_encoder(cfg):
    # 加载VGG16预训练
    base_model = VGG16(weights='imagenet', include_top=True) #'imagenet''./pretrain_model/vgg16_weights.h5'
    # base_model.summary()
    # encoder
    encoder = Sequential()
    for i in range(12):
        base_model.layers[i].trainable = False
        encoder.add(base_model.layers[i])
    encoder.add(Conv2D(512, (3, 3), padding='VALID', name='encoder_conv2D1',kernel_initializer=heinit,bias_initializer=zeroinit))
    encoder.add(BatchNormalization())
    encoder.add(ELU())
    encoder.add(Conv2D(512, (3, 3), padding='VALID', name='encoder_conv2D2',kernel_initializer=heinit,bias_initializer=zeroinit))
    encoder.add(BatchNormalization())
    encoder.add(ELU())
    encoder.add(MaxPooling2D((3, 3), strides=(3, 3), name='encoder_maxpool1'))
    encoder.add(Conv2D(256, (1, 1), padding='VALID', name='encoder_conv2D3',kernel_initializer=heinit,bias_initializer=zeroinit))
    encoder.add(BatchNormalization())
    encoder.add(ELU())
    return encoder
    
    
def gen_encoder_nonorm(cfg):
    # 加载VGG16预训练
    base_model = VGG16(weights='./pretrain_model/vgg16_weights.h5', include_top=True) #'imagenet'
    # base_model.summary()
    # encoder
    encoder = Sequential()
    for i in range(12):
        base_model.layers[i].trainable = False
        encoder.add(base_model.layers[i])
    encoder.add(Conv2D(512, (3, 3), padding='VALID', name='encoder_conv2D1',kernel_initializer=heinit,bias_initializer=zeroinit))
    #encoder.add(BatchNormalization())
    encoder.add(ELU())
    #encoder.add(Dropout(0.1))
    encoder.add(Conv2D(512, (3, 3), padding='VALID', name='encoder_conv2D2',kernel_initializer=heinit,bias_initializer=zeroinit))
    #encoder.add(BatchNormalization())
    encoder.add(ELU())
    encoder.add(MaxPooling2D((3, 3), strides=(3, 3), name='encoder_maxpool1'))
    #encoder.add(Dropout(0.05))
    encoder.add(Conv2D(256, (1, 1), padding='VALID', name='encoder_conv2D3',kernel_initializer=heinit,bias_initializer=zeroinit))
    #encoder.add(BatchNormalization())
    encoder.add(ELU())
    #encoder.add(Dropout(0.05))
    return encoder
    
def gen_encoder_nonorm_f(cfg):
    # 加载VGG16预训练
    base_model = VGG16(weights='./pretrain_model/vgg16_weights.h5', include_top=True) #'imagenet'
    # base_model.summary()
    # encoder
    encoder = Sequential()
    for i in range(12):
        base_model.layers[i].trainable = False
        encoder.add(base_model.layers[i])
    encoder.add(Conv2D(512, (1, 1), padding='VALID', name='encoder_conv2D1',kernel_initializer=heinit,bias_initializer=zeroinit))
    #encoder.add(BatchNormalization())
    encoder.add(ELU())
    #encoder.add(Dropout(0.1))
    encoder.add(Conv2D(256, (3, 3), padding='VALID', name='encoder_conv2D2',kernel_initializer=heinit,bias_initializer=zeroinit))
    #encoder.add(BatchNormalization())
    encoder.add(ELU())
    encoder.add(MaxPooling2D((4, 4), strides=(4, 4), name='encoder_maxpool1')) #have porb
    #encoder.add(Dropout(0.1))
    encoder.add(Conv2D(128, (3, 3), padding='VALID', name='encoder_conv2D3',kernel_initializer=heinit,bias_initializer=zeroinit))
    #encoder.add(BatchNormalization())
    encoder.add(ELU())
    #encoder.add(Dropout(0.1))
    return encoder