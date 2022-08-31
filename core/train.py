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
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras import backend as kb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv3DTranspose, MaxPooling2D,MaxPooling3D, \
    concatenate, Input, Reshape, BatchNormalization, ELU, ReLU, Conv3D, LeakyReLU

import utils.binvox_rw
import utils.datasetloader
import utils.preprocessing
import utils.data_transforms

import utils_origin.data_loaders

import model.encoder
import model.decoder
import model.refiner
import model.merger
import model.loss
import model.edcombiner



def train_net(cfg):
    # load dataset
    #train_loader = utils.datasetloader.dataloader(cfg,"train")
    #train_loader = train_loader[:3]  #for test
    #file=open('log/log.txt',mode='w')
    #file.write(str(train_loader))
    #print(train_loader)
    #input()
    #batch_num = int(len(train_loader) / cfg.CONST.BATCH_SIZE)
    #train_gen = utils.datasetloader.generator_trainv2(cfg)
    #next(train_gen)
    #print("fff")
    #input()
    train_gen = utils.datasetloader.generator_trainv3(cfg,'test')
    batch_num = next(train_gen)
    print(batch_num)
    val_gen = utils.datasetloader.valdata_gen()
    val_dataset = tf.data.Dataset.from_generator(utils.datasetloader.valdata_gen, output_types=(tf.float32, tf.float32), output_shapes=None, args=None,output_signature=None)
    val_dataset = val_dataset.batch(cfg.CONST.BATCH_SIZE)
    #print(list(val_dataset.take(1))[0][1].shape)
    #print(list(val_dataset.take(1))[1].shape)
    #input()
    #val_loader = utils.datasetloader.dataloader(cfg, "val")
    #val_loader = val_loader[:3]    #for test
    #val_batch_num = int(len(val_loader) / cfg.CONST.BATCH_SIZE)
    #val_loader = utils.datasetloader.dataloader_permute(val_loader)
    #val_dataset = tf.data.Dataset.from_tensor_slices(val_loader)
    #val_dataset = val_dataset.map(
    #    lambda x, y: tf.py_function(utils.datasetloader.mapload_funv2, inp=[x, y, 'val'],
    #                                Tout=[tf.float32, tf.float32]),num_parallel_calls=tf.data.AUTOTUNE)#
    #val_dataset = val_dataset.batch(cfg.CONST.BATCH_SIZE)
    #val_dataset=val_dataset.repeat()
    
    
    
    encoder=model.encoder.gen_encoder_nonorm(cfg)
    #encoder.summary()
    decoder=model.decoder.gen_decoder_nonorm(cfg)
    #decoder.summary()
    
    if cfg.CONST.N_VIEWS_RENDERING == 1:
        if cfg.CONST.TRAIN_REFINER == 0 :
            Pix2Vox = model.edcombiner.add_combiner(cfg,encoder,decoder)
        else:
            Pix2Vox = model.refiner.add_refiner_nonorm(cfg, encoder, decoder)
            Pix2Vox.load_weights('./rbest.h5')
            Pix2Vox1 = model.edcombiner.add_combiner(cfg,encoder,decoder)
            Pix2Vox1.load_weights('./edbest.h5')
            for layerindex in range(len(Pix2Vox1.layers)):
                Pix2Vox.layers[layerindex].set_weights(Pix2Vox1.layers[layerindex].get_weights())
                #Pix2Vox.layers[layerindex].trainable = False
    else:
        Pix2Vox1 = model.refiner.add_refiner_nonorm(cfg, encoder, decoder)
        Pix2Vox1.load_weights('./rbest6550.h5')
        '''
        Pix2Vox2 = model.edcombiner.add_combiner(cfg,encoder,decoder)
        Pix2Vox2.load_weights('./edbest.h5')
        for layerindex in range(len(Pix2Vox2.layers)):
            Pix2Vox1.layers[layerindex].set_weights(Pix2Vox2.layers[layerindex].get_weights())
            Pix2Vox1.layers[layerindex].trainable = False
        '''
        lenencoderdecoder=len(encoder.layers)+len(decoder.layers)+1
        Pix2Vox = model.merger.add_mergerv2(cfg, Pix2Vox1,lenencoderdecoder)
        del(Pix2Vox1)
        #del(Pix2Vox2)
    
    del(encoder)
    del(decoder)
    Pix2Vox.summary(line_length=110)
    print(Pix2Vox.outputs)
    print("check the model")
    input()


    #Pix2Vox = model.merger.add_merger(cfg,Pix2Vox1)
    #del(Pix2Vox1)
    #Pix2Vox.summary(line_length=200)
    #input()
    #2.0
    def IoU(y_true, y_pred):
        one = tf.ones_like(y_pred)  # 生成与a大小一致的值全部为1的矩阵
        zero = tf.zeros_like(y_pred)
        predict_volume = tf.where(y_pred <= cfg.TRAIN.VOXEL_THRESH, x=zero, y=one)
        num = tf.reduce_sum(predict_volume * y_true, (0, 1, 2, 3))
        dec = tf.reduce_sum(tf.where((predict_volume + y_true) < 1, x=zero, y=one), (0, 1, 2, 3))
        return num/dec
    optimizer = optimizers.Adam(lr=cfg.TRAIN.LEARNING_RATE, beta_1=cfg.TRAIN.BETAS[0], beta_2=cfg.TRAIN.BETAS[1], epsilon=1e-8)
    #optimizer = optimizers.SGD(lr=cfg.TRAIN.ENCODER_LEARNING_RATE)
    #Pix2Vox.compile(loss=[model.loss.Pix2Vox_loss_fn,model.loss.Pix2Vox_loss_fn], optimizer=optimizer,metrics=[IoU])
    Pix2Vox.compile(loss=model.loss.Pix2Vox_loss_fn, optimizer=optimizer,metrics=[IoU])
    tensorboard = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None, embeddings_data=None, update_freq=cfg.TRAIN.SAVE_FREQ)
    
    def scheduler(epoch):
        if epoch>=cfg.TRAIN.LR_MILESTONES[0]:
            print("lr {}".format(cfg.TRAIN.LEARNING_RATE*cfg.TRAIN.GAMMA))
            return cfg.TRAIN.LEARNING_RATE*cfg.TRAIN.GAMMA
        else:
            print("lr {}".format(cfg.TRAIN.LEARNING_RATE))
            return  cfg.TRAIN.LEARNING_RATE
    reduce_lr = LearningRateScheduler(scheduler)
    #best_weight_path1='./log/weights-impovement-epoch{epoch:02d}--val_R_IoU{val_R_IoU:.4f}.hdf5'
    #best_weight_path1='./log/weights-impovement-epoch{epoch:02d}--val_IoU{val_IoU:.4f}.hdf5'
    best_weight_path1='./log/weights-impovement-epoch{epoch:02d}--IoU{IoU:.4f}.hdf5'
    checkpoint = ModelCheckpoint(best_weight_path1,
                                 monitor='val_IoU', save_weights_only=True, verbose=1, save_best_only=False,mode='max', save_freq='epoch')
    best_weight_path='./log/best.h5'
    if os.path.exists(best_weight_path):
        Pix2Vox.load_weights(best_weight_path)
        # 若成功加载前面保存的参数，输出下列信息
        print("checkpoint_loaded")
    '''
    res = Pix2Vox.evaluate(val_dataset)
    file=open('test_before_train.txt',mode='a+')
    file.write(str(res))
    file.close()
    print(res)
    '''
    print('start trainning...')
    '''
    Pix2Vox.fit(train_gen, steps_per_epoch=batch_num,
                epochs=cfg.TRAIN.NUM_EPOCHES, validation_data=val_dataset,
                callbacks=[checkpoint, tensorboard, reduce_lr], initial_epoch=cfg.CONST.INITIAL_EPOCH,use_multiprocessing=False)  # ，batch_num ,val_batch_num,validation_steps=2 validation_steps=val_batch_num
    '''
    Pix2Vox.fit(train_gen, steps_per_epoch=batch_num,
                epochs=cfg.TRAIN.NUM_EPOCHES,
                callbacks=[checkpoint, tensorboard, reduce_lr], initial_epoch=cfg.CONST.INITIAL_EPOCH,use_multiprocessing=False)  # ，batch_num ,val_batch_num
    
    #input()

