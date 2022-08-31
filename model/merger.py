# Developed by Mingzhe Piao 
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras.backend as k
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv3DTranspose, MaxPooling2D,MaxPooling3D, \
    concatenate, Input, Reshape, BatchNormalization, ELU, ReLU, Conv3D, LeakyReLU,Lambda, Concatenate, Softmax
from tensorflow.keras import initializers
zeroinit=initializers.Zeros()
oneinit=initializers.Ones()
heinit=initializers.he_normal()
normalinit=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
from config import cfg

    
def add_mergerv2(cfg,model,lenencoderdecoder):
    lenencoderdecoder= lenencoderdecoder+1
    #print("auto_encoder")
    #auto_encoder.summary(line_length=110)
    #input()
    #auto_encoder_inputs = []
    #merger
    m1 = Conv3D(16, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=zeroinit,bias_initializer=zeroinit, name='merger_start')
    m2 = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)
    m3 = Conv3D(8, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=zeroinit,bias_initializer=zeroinit)
    m4 = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)
    m5 = Conv3D(4, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=zeroinit,bias_initializer=zeroinit)
    m6 = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)
    m7 = Conv3D(2, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=zeroinit,bias_initializer=zeroinit)
    m8 = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)
    m9 = Conv3D(1, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=zeroinit,bias_initializer=zeroinit)
    m10 = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE,name='merger_end')
    edcx = []
    edx = []
    auto_encoder_input = Input(shape=(cfg.CONST.N_VIEWS_RENDERING,cfg.CONST.IMG_H, cfg.CONST.IMG_W, 3))
    #x = Lambda(lambda input:tf.split(input,cfg.CONST.N_VIEWS_RENDERING,1))(auto_encoder_input)
    inputs=[]
    #for i in range(cfg.CONST.N_VIEWS_RENDERING):
    #    inputs.append(Lambda(lambda input: input[:, i,:,:,:],name='input'+str(i))(auto_encoder_input))
    print("view establish:")
    print(cfg.CONST.N_VIEWS_RENDERING)
    if cfg.CONST.N_VIEWS_RENDERING==2:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==3:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==4:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==5:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==6:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==7:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==8:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==10:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 8,:,:,:],name='input9')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 9,:,:,:],name='input10')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==12:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 8,:,:,:],name='input9')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 9,:,:,:],name='input10')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 10,:,:,:],name='input11')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 11,:,:,:],name='input12')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==16:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 8,:,:,:],name='input9')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 9,:,:,:],name='input10')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 10,:,:,:],name='input11')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 11,:,:,:],name='input12')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 12,:,:,:],name='input13')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 13,:,:,:],name='input14')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 14,:,:,:],name='input15')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 15,:,:,:],name='input16')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==20:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 8,:,:,:],name='input9')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 9,:,:,:],name='input10')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 10,:,:,:],name='input11')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 11,:,:,:],name='input12')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 12,:,:,:],name='input13')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 13,:,:,:],name='input14')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 14,:,:,:],name='input15')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 15,:,:,:],name='input16')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 16,:,:,:],name='input17')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 17,:,:,:],name='input18')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 18,:,:,:],name='input19')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 19,:,:,:],name='input20')(auto_encoder_input))
    else:
        print("unsupported views")
    for i in range(cfg.CONST.N_VIEWS_RENDERING):
        #inp = Input(shape=(cfg.CONST.IMG_H, cfg.CONST.IMG_W, 3))
        #auto_encoder_inputs.append(inp)
        #x[i] = Lambda(lambda input: tf.squeeze(input, axis=1))(x[i])
        #x = Lambda(lambda input: input[:, i,:,:,:])(auto_encoder_input)
        x=inputs[i]
        # connect encoder decoder and refiner
        flag=False
        for i in range(lenencoderdecoder-1):
            if model.layers[i].name[0]=='e':
                flag=True
            if flag:
                #model.layers[i].trainable = True
                model.layers[i].trainable = False
            x = model.layers[i](x)
        rx = x
        x = model.layers[lenencoderdecoder-1](x)
        #model.layers[lenencoderdecoder-1].trainable = True
        model.layers[lenencoderdecoder-1].trainable = False
        cx = x
        edcx.append(cx)
        x = Concatenate(axis=4)([rx, x])
        #merger
        x=m1(x)
        x=m2(x)
        x=m3(x)
        x=m4(x)
        x=m5(x)
        x=m6(x)
        x=m7(x)
        x=m8(x)
        x=m9(x)
        x=m10(x)
        edx.append(x)
    if(cfg.CONST.N_VIEWS_RENDERING != 1):
        flx = concatenate(edx)
        cflx = concatenate(edcx)
    else:
        flx=edx[0]
        cflx=edcx[0]
    flx = Softmax(axis=4)(flx)
    flx = flx * cflx
    flx = Lambda(lambda x:k.expand_dims(k.sum(x,4),4),name='EDn')(flx)
    
    x= flx
    x32 = x
    #refiner_input = Input(shape=(32, 32, 32, 1))
    #x = refiner_input
    x = Conv3D(32, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit,name='refiner_start')(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x16 = MaxPooling3D((2, 2, 2))(x)
    x = x16
    x = Conv3D(64, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x8 = MaxPooling3D((2, 2, 2))(x)
    x = x8
    x = Conv3D(128, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
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
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(32, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x + x8)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(1, (4, 4, 4), padding='SAME', strides=(2, 2, 2),activation='sigmoid',kernel_initializer=heinit,bias_initializer=zeroinit)(x + x16)
    x = (x+x32)*0.5
    x = Lambda(lambda x: x, name='R')(x)
    
    final_auto_encoder = Model(inputs=auto_encoder_input, outputs=[x32, x])
    #final_auto_encoder = Model(inputs=auto_encoder_input, outputs=[x])
    #test
    #final_auto_encoder.summary(line_length=110)
    #input()
    refiner_index = 0
    for refiner_point in range(len(final_auto_encoder.layers)):
        if(final_auto_encoder.layers[refiner_point].name=='refiner_start'):
            refiner_index = refiner_point
            break
    #update weight of refiner
    for j in range(lenencoderdecoder,len(model.layers)):
        #final_auto_encoder.layers[j-29+cfg.CONST.N_VIEWS_RENDERING].set_weights(model.layers[j].get_weights())
        final_auto_encoder.layers[j-lenencoderdecoder+refiner_index].set_weights(model.layers[j].get_weights())
        final_auto_encoder.layers[j-lenencoderdecoder+refiner_index].trainable = False
        
    
    #final_auto_encoder = Model(inputs=auto_encoder_input, outputs=[flx])
    return final_auto_encoder
    
def add_aver(cfg,model,lenencoderdecoder):
    lenencoderdecoder= lenencoderdecoder+1
    #print("auto_encoder")
    #auto_encoder.summary(line_length=110)
    #input()
    #auto_encoder_inputs = []
    #merger
    edcx = []
    auto_encoder_input = Input(shape=(cfg.CONST.N_VIEWS_RENDERING,cfg.CONST.IMG_H, cfg.CONST.IMG_W, 3))
    #x = Lambda(lambda input:tf.split(input,cfg.CONST.N_VIEWS_RENDERING,1))(auto_encoder_input)
    inputs=[]
    #for i in range(cfg.CONST.N_VIEWS_RENDERING):
    #    inputs.append(Lambda(lambda input: input[:, i,:,:,:],name='input'+str(i))(auto_encoder_input))
    '''
    inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
    
    inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
    
    inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
    
    inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
    
    inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
    
    inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
    
    inputs.append(Lambda(lambda input: input[:, 8,:,:,:],name='input9')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 9,:,:,:],name='input10')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 10,:,:,:],name='input11')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 11,:,:,:],name='input12')(auto_encoder_input))
    
    inputs.append(Lambda(lambda input: input[:, 12,:,:,:],name='input13')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 13,:,:,:],name='input14')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 14,:,:,:],name='input15')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 15,:,:,:],name='input16')(auto_encoder_input))
    
    inputs.append(Lambda(lambda input: input[:, 16,:,:,:],name='input17')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 17,:,:,:],name='input18')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 18,:,:,:],name='input19')(auto_encoder_input))
    inputs.append(Lambda(lambda input: input[:, 19,:,:,:],name='input20')(auto_encoder_input))
    '''
    print("view establish:")
    print(cfg.CONST.N_VIEWS_RENDERING)
    if cfg.CONST.N_VIEWS_RENDERING==2:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==3:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==4:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==5:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==6:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==7:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==8:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==10:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 8,:,:,:],name='input9')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 9,:,:,:],name='input10')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==12:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 8,:,:,:],name='input9')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 9,:,:,:],name='input10')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 10,:,:,:],name='input11')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 11,:,:,:],name='input12')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==16:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 8,:,:,:],name='input9')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 9,:,:,:],name='input10')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 10,:,:,:],name='input11')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 11,:,:,:],name='input12')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 12,:,:,:],name='input13')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 13,:,:,:],name='input14')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 14,:,:,:],name='input15')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 15,:,:,:],name='input16')(auto_encoder_input))
    elif cfg.CONST.N_VIEWS_RENDERING==20:
        inputs.append(Lambda(lambda input: input[:, 0,:,:,:],name='input1')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 1,:,:,:],name='input2')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 2,:,:,:],name='input3')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 3,:,:,:],name='input4')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 4,:,:,:],name='input5')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 5,:,:,:],name='input6')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 6,:,:,:],name='input7')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 7,:,:,:],name='input8')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 8,:,:,:],name='input9')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 9,:,:,:],name='input10')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 10,:,:,:],name='input11')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 11,:,:,:],name='input12')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 12,:,:,:],name='input13')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 13,:,:,:],name='input14')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 14,:,:,:],name='input15')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 15,:,:,:],name='input16')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 16,:,:,:],name='input17')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 17,:,:,:],name='input18')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 18,:,:,:],name='input19')(auto_encoder_input))
        inputs.append(Lambda(lambda input: input[:, 19,:,:,:],name='input20')(auto_encoder_input))
    else:
        print("unsupported views")
    for i in range(cfg.CONST.N_VIEWS_RENDERING):
        #inp = Input(shape=(cfg.CONST.IMG_H, cfg.CONST.IMG_W, 3))
        #auto_encoder_inputs.append(inp)
        #x[i] = Lambda(lambda input: tf.squeeze(input, axis=1))(x[i])
        #x = Lambda(lambda input: input[:, i,:,:,:])(auto_encoder_input)
        x=inputs[i]
        # connect encoder decoder and refiner
        flag=False
        for i in range(lenencoderdecoder-1):
            if model.layers[i].name[0]=='e':
                flag=True
            if flag:
                model.layers[i].trainable = True
            x = model.layers[i](x)
        rx = x
        x = model.layers[lenencoderdecoder-1](x)
        model.layers[lenencoderdecoder-1].trainable = True
        cx = x
        edcx.append(cx)
    if(cfg.CONST.N_VIEWS_RENDERING != 1):
        cflx = concatenate(edcx)
    else:
        cflx=edcx[0]
    flx = cflx/cfg.CONST.N_VIEWS_RENDERING
    flx = Lambda(lambda x:k.expand_dims(k.sum(x,4),4),name='EDn')(flx)
    x= flx
    x32 = x
    #refiner_input = Input(shape=(32, 32, 32, 1))
    #x = refiner_input
    x = Conv3D(32, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit,name='refiner_start')(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x16 = MaxPooling3D((2, 2, 2))(x)
    x = x16
    x = Conv3D(64, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x8 = MaxPooling3D((2, 2, 2))(x)
    x = x8
    x = Conv3D(128, (4, 4, 4), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
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
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(32, (4, 4, 4), padding='SAME', strides=(2, 2, 2),kernel_initializer=heinit,bias_initializer=zeroinit)(x + x8)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(1, (4, 4, 4), padding='SAME', strides=(2, 2, 2),activation='sigmoid',kernel_initializer=heinit,bias_initializer=zeroinit)(x + x16)
    x = (x+x32)*0.5
    x = Lambda(lambda x: x, name='R')(x)
    
    final_auto_encoder = Model(inputs=auto_encoder_input, outputs=[x32, x])
    #final_auto_encoder = Model(inputs=auto_encoder_input, outputs=[x])
    #test
    #final_auto_encoder.summary(line_length=110)
    #input()
    refiner_index = 0
    for refiner_point in range(len(final_auto_encoder.layers)):
        if(final_auto_encoder.layers[refiner_point].name=='refiner_start'):
            refiner_index = refiner_point
            break
    #update weight of refiner
    for j in range(lenencoderdecoder,len(model.layers)):
        #final_auto_encoder.layers[j-29+cfg.CONST.N_VIEWS_RENDERING].set_weights(model.layers[j].get_weights())
        final_auto_encoder.layers[j-lenencoderdecoder+refiner_index].set_weights(model.layers[j].get_weights())
        #final_auto_encoder.layers[j-lenencoderdecoder+refiner_index].trainable = False
    #final_auto_encoder = Model(inputs=auto_encoder_input, outputs=[flx])
    return final_auto_encoder
    
def add_merger_f(cfg,model,lenencoderdecoder):
    # connect encoder decoder and refiner
    input_x = Input(shape=(cfg.CONST.IMG_W, cfg.CONST.IMG_H, 3))
    x = input_x
    for i in range(lenencoderdecoder-1):
        x = model.layers[i](x)

    rx = x
    x = model.layers[lenencoderdecoder-1](x)
    cx = x
    x = Concatenate(axis=4)([rx, x])
    #merger
    x = Conv3D(16, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x = Conv3D(8, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x = Conv3D(4, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x = Conv3D(2, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)
    x = Conv3D(1, (3, 3, 3), padding='SAME', strides=1,kernel_initializer=heinit,bias_initializer=zeroinit)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=cfg.NETWORK.LEAKY_VALUE)(x)

    auto_encoder = Model(inputs=input_x, outputs=[x, cx])

    #auto_encoder_inputs = []
    auto_encoder_models1 = []
    auto_encoder_models2 = []
    auto_encoder_input = Input(shape=(cfg.CONST.N_VIEWS_RENDERING,cfg.CONST.IMG_H, cfg.CONST.IMG_W, 3))
    #x = Lambda(lambda input:tf.split(input,cfg.CONST.N_VIEWS_RENDERING,1))(auto_encoder_input)

    for i in range(cfg.CONST.N_VIEWS_RENDERING):
        #inp = Input(shape=(cfg.CONST.IMG_H, cfg.CONST.IMG_W, 3))
        #auto_encoder_inputs.append(inp)
        #x[i] = Lambda(lambda input: tf.squeeze(input, axis=1))(x[i])
        x = Lambda(lambda input: input[:, i])(auto_encoder_input)
        res = auto_encoder(x)
        auto_encoder_models1.append(res[0])
        auto_encoder_models2.append(res[1])
    flx = concatenate(auto_encoder_models1)
    cflx = concatenate(auto_encoder_models2)
    flx = Softmax(axis=4)(flx)
    flx = flx * cflx
    flx = Lambda(lambda x:k.expand_dims(k.sum(x,4),4),name='ED')(flx)
    final_auto_encoder = Model(inputs=auto_encoder_input, outputs=[flx])
    #test
    final_auto_encoder.summary()
    #input()
    input()
    return final_auto_encoder