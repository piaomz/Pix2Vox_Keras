# -*- coding: utf-8 -*-
#
# Developed by Mingzhe Piao 
# using Haozhe Xie <cshzxie@gmail.com> works

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/ShapeNet.json'
# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH  = './datasets/PascalShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/home/userPiaoMingze/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
# __C.DATASETS.SHAPENET.RENDERING_PATH      = '/home/userPiaoMingze/ShapeNet/PascalShapeNetRendering/%s/%s/render_%04d.jpg'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/home/userPiaoMingze/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
#__C.DATASETS.PASCAL3D                       = edict()
#__C.DATASETS.PASCAL3D.TAXONOMY_FILE_PATH    = './datasets/Pascal3D.json'
#__C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/home/piaomz/Pix2Vox/datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
#__C.DATASETS.PASCAL3D.RENDERING_PATH        = '/home/piaomz/Pix2Vox/datasets/Images/%s_imagenet/%s.JPEG'
#__C.DATASETS.PASCAL3D.VOXEL_PATH            = '/home/piaomz/Pix2Vox/datasets/CAD/%s/%02d.binvox'
#__C.DATASETS.PIX3D                          = edict()
#__C.DATASETS.PIX3D.TAXONOMY_FILE_PATH       = './datasets/Pix3D.json'
#__C.DATASETS.PIX3D.ANNOTATION_PATH          = '/home/piaomz/Pix2Vox/datasets/Pix3D/pix3d.json'
#__C.DATASETS.PIX3D.RENDERING_PATH           = '/home/piaomz/Pix2Vox/datasets/Pix3D/img/%s/%s.%s'
#__C.DATASETS.PIX3D.VOXEL_PATH               = '/home/piaomz/Pix2Vox/datasets/Pix3D/model/%s/%s/%s.binvox'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNet'
__C.DATASET.TEST_DATASET                    = 'ShapeNet'
# __C.DATASET.TEST_DATASET                  = 'Pascal3D'
# __C.DATASET.TEST_DATASET                  = 'Pix3D'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '-1'
__C.CONST.GPU_MEM                           = 7500
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 16
__C.CONST.N_VIEWS_RENDERING                 = 20  
__C.CONST.CROP_IMG_W                        = 128       
__C.CONST.CROP_IMG_H                        = 128       
__C.CONST.USE_F                             = 0
__C.CONST.TRAIN_REFINER                     = 0
__C.CONST.INITIAL_EPOCH                     = 0
#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'
__C.DIR.RANDOM_BG_PATH                      = '/home/userPiaoMingze/SUN2012/Images/*/*/*.jpg'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = True

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 1             # number of data workers: unavailable setting
__C.TRAIN.NUM_EPOCHES                       = 500
__C.TRAIN.BRIGHTNESS                        = .5
__C.TRAIN.CONTRAST                          = .5
__C.TRAIN.SATURATION                        = .5
__C.TRAIN.JEPG_QUALITY                      = 75
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: just adam
__C.TRAIN.LEARNING_RATE                     = 1e-4 
#__C.TRAIN.LEARNING_RATE                     = 1e-5
#__C.TRAIN.LEARNING_RATE                     = 1e-4
__C.TRAIN.LR_MILESTONES                     = [50]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 1            
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False
__C.TRAIN.VOXEL_THRESH                       = 0.3

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.05,.1,.15, .2, .25,.3,.35, .4,.45, .5]
