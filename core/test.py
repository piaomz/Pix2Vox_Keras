import os
import random
import json
from datetime import datetime as dt
import numpy as np
import cv2
print("123")
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as kb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv3DTranspose, MaxPooling2D,MaxPooling3D, \
    concatenate, Input, Reshape, BatchNormalization, ELU, ReLU, Conv3D, LeakyReLU, Softmax
print("223")
import utils.binvox_rw
import utils.datasetloader
import utils.preprocessing
import model.encoder
import model.decoder
import model.refiner
import model.edcombiner
import model.loss
import model.merger

def test_net(cfg):
    encoder=model.encoder.gen_encoder_nonorm(cfg)
    #encoder.summary()
    decoder=model.decoder.gen_decoder_nonorm(cfg)
    #decoder.summary()
    print("establish networks")
    lenencoderdecoder=len(encoder.layers)+len(decoder.layers)+1
    Pix2VoxR = model.refiner.add_refiner_nonorm(cfg, encoder, decoder)
    #Pix2VoxR.load_weights('./rbest.h5')
    Pix2VoxR.load_weights('./rbest6550.h5')
    Pix2VoxED = model.edcombiner.add_combiner(cfg,encoder,decoder)
    #Pix2VoxED.load_weights('./edbest.h5')
    #for i in range(len(Pix2VoxED.layers)):
    #     Pix2VoxR.layers[i].set_weights(Pix2VoxED.layers[i].get_weights())
    '''
    Pix2VoxM = model.merger.add_mergerv2(cfg, Pix2VoxR,lenencoderdecoder)
    Pix2VoxM2 = model.merger.add_mergerv2(cfg, Pix2VoxR,lenencoderdecoder)
    Pix2VoxM2.load_weights('./mbest.h5')
    mstart1=0
    mend1=0
    mstart2=0
    mend2=0
    for i in range(len(Pix2VoxM.layers)):
        if Pix2VoxM.layers[i].name=='merger_start':
            mstart1=i
        if Pix2VoxM.layers[i].name=='merger_end':
            mend1=i
    for i in range(len(Pix2VoxM2.layers)):
        if Pix2VoxM.layers[i].name=='merger_start':
            mstart2=i
        if Pix2VoxM.layers[i].name=='merger_end':
            mend2=i
    for i in range(mstart1, mend1+1):
        Pix2VoxM.layers[i].set_weights(Pix2VoxM2.layers[mstart2].get_weights())
        mstart2=mstart2+1
    del(Pix2VoxM2)
    #Pix2VoxM2.load_weights('./mbest.h5')
    '''
    view_number=[12,2,3,4,5,6,7,8,10,16,20]
    for viewnum in view_number:
        print(viewnum)
        cfg.CONST.N_VIEWS_RENDERING=viewnum
        #Pix2VoxA = model.merger.add_aver(cfg, Pix2VoxR,lenencoderdecoder)
        #Pix2VoxM = model.merger.add_mergerv2(cfg, Pix2VoxR,lenencoderdecoder,viewnum)
        #Pix2VoxM.load_weights('./mbest.h5')
        print("load weight")
        #del(encoder)
        #del(decoder)
        #Pix2VoxFull = model.refiner.add_refiner_m(cfg,Pix2VoxM,Pix2VoxR,lenencoderdecoder)
        #Pix2Vox=Pix2VoxR
        for mergerindex in range(1):
            if mergerindex==0:
                #del(Pix2Vox)
                Pix2VoxR.load_weights('./rbest6550.h5')
                Pix2Vox=model.merger.add_mergerv2(cfg, Pix2VoxR,lenencoderdecoder)
                Pix2Vox.load_weights('./log/weights-impovement-epoch05--val_IoU0.6943.hdf5')
            else:
                Pix2VoxR.load_weights('./rbest6550.h5')
                Pix2Vox=model.merger.add_aver(cfg, Pix2VoxR,lenencoderdecoder)
            Pix2Vox.summary()
            print(Pix2Vox.outputs)
            print(viewnum)
            print(mergerindex)
            #input()
            '''
            if cfg.CONST.N_VIEWS_RENDERING != 1:
                Pix2Vox = model.merger.add_merger(cfg,Pix2Vox1)
                del(Pix2Vox1)
                Pix2Vox1 = Pix2Vox
            Pix2Vox1.summary(line_length=200)
            #input()
            '''
            #2.0
            #Pix2VoxN= model.merger.add_merger_refiner(cfg,Pix2Vox)
            def IoU_test0(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[0], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            def IoU_test1(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[1], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            def IoU_test2(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[2], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            def IoU_test3(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[3], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            def IoU_test4(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[4], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            def IoU_test5(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[5], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            def IoU_test6(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[6], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            def IoU_test7(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[7], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            def IoU_test8(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[8], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            def IoU_test9(y_true, y_pred):
                IoU=[]
                one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
                zero = tf.zeros_like(y_pred)
                predict_volume = tf.where(y_pred <= cfg.TEST.VOXEL_THRESH[9], x=zero, y=one)
                num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
                dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
                return num/dec
            Pix2Vox.compile(loss=[model.loss.Pix2Vox_loss_fn, model.loss.Pix2Vox_loss_fn],  metrics=[IoU_test0,IoU_test1,IoU_test2,IoU_test3,IoU_test4,IoU_test5,IoU_test6,IoU_test7,IoU_test8,IoU_test9])
            #Pix2Vox.compile(loss=[model.loss.Pix2Vox_loss_fn, model.loss.Pix2Vox_loss_fn],  metrics=[IoU_test0,IoU_test1,IoU_test2,IoU_test3,IoU_test4,IoU_test5,IoU_test6])
            '''
            optimizer = optimizers.Adam(lr=cfg.TRAIN.ENCODER_LEARNING_RATE, beta_1=cfg.TRAIN.BETAS[0], beta_2=cfg.TRAIN.BETAS[1], epsilon=1e-8)
            Pix2Vox.compile(loss=[model.loss.Pix2Vox_loss_fn,model.loss.Pix2Vox_loss_fn], optimizer=optimizer,metrics=[IoU])
            tensorboard = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=False, write_grads=False,
                                        write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                        embeddings_metadata=None, embeddings_data=None, update_freq=cfg.TRAIN.SAVE_FREQ)
        
            best_weight_path='./log/best.h5'
            checkpoint = ModelCheckpoint(best_weight_path,
                                         monitor='loss', save_weights_only=True, verbose=1, save_best_only=True, save_freq=cfg.TRAIN.SAVE_FREQ)
            '''
            datasetpath=cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH
            #classname=['airplane']
            '''
            for i in range(13):
                cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = datasetpath[:-5]+str(i)+'.json'
                test_dataset = tf.data.Dataset.from_generator(utils.datasetloader.testdata_gen, output_types=(tf.float32, tf.float32), output_shapes=None, args=None)
                test_dataset = test_dataset.batch(1)
                res = Pix2Vox.evaluate(test_dataset)
                file=open('testres/test'+str(i)+'.txt',mode='a+')
                file.write(str(res))
                file.close()
                print(res)
               ''' 
            cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH=datasetpath
            test_dataset = tf.data.Dataset.from_generator(utils.datasetloader.testdata_gen, output_types=(tf.float32, tf.float32), output_shapes=None, args=None)
            test_dataset = test_dataset.batch(1)
            res = Pix2Vox.evaluate(test_dataset)
            file=open('testres/test'+str(cfg.CONST.N_VIEWS_RENDERING)+'.txt',mode='a+')
            file.write(str(res))
            file.close()
            print(res)
        #input()
        #resR = np.squeeze(res[1],axis=0)
        #resR = np.squeeze(resR,axis=3)
        #resR = np.where(resR > 0.5, 1, 0)
        del(Pix2Vox)
    '''
    print('start trainning...')
    #Pix2Vox.fit(utils.datasetloader.generator(cfg, train_loader,'train'), steps_per_epoch=30,
    #            epochs=cfg.TRAIN.NUM_EPOCHES,validation_data=utils.datasetloader.generator_val(cfg, val_loader,'val'), validation_steps=10,callbacks=[checkpoint,tensorboard])  #，batch_num
    Pix2Vox.fit(train_dataset, steps_per_epoch=batch_num,
                epochs=cfg.TRAIN.NUM_EPOCHES, validation_data=val_dataset,
                validation_steps=val_batch_num, callbacks=[checkpoint, tensorboard])  # ，batch_num ,val_batch_num,validation_steps=2

    #input()
    '''
