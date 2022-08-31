import os
import numpy as np
import cv2
import random
import json
from datetime import datetime as dt
import torch
import tensorflow as tf

import utils_origin.binvox_visualization
import utils_origin.data_loaders
#import utils_origin.data_transforms
#import utils.network_utils

import utils.binvox_rw
import utils.data_transforms
from config import cfg


def generator_train(cfg):
    #backupdataset= dataset
    #epochdataset = dataset
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE,random_bg_folder_path=cfg.DIR.RANDOM_BG_PATH),#
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
    ])
    
    while True:
        train_loader = utils.datasetloader.dataloader(cfg,"train")
        sizeofdataset=len(train_loader)
        random.shuffle(train_loader)
        train_loader_permute = utils.datasetloader.dataloader_permute(train_loader)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_loader_permute)
        train_dataset.shuffle(sizeofdataset)
        train_dataset = train_dataset.map(lambda x, y: tf.py_function(utils.datasetloader.mapload_funv2, inp=[x, y,'train'], Tout=[tf.float32, tf.float32]),num_parallel_calls=tf.data.AUTOTUNE)#,num_parallel_calls=tf.data.AUTOTUNE
        #epochdataset = epochdataset.map(utils.preprocessing.preprocessing_train,num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(cfg.CONST.BATCH_SIZE)
        for item in train_dataset:
            #print(item)
            x,y= item
            #print(x.numpy())
            if(cfg.CONST.N_VIEWS_RENDERING == 1):
                x=transforms(x.numpy())
            else:
                pass  #wait for programming
            yield x,y
        del(train_dataset)
def generator_trainv2(cfg):
    #backupdataset= dataset
    #epochdataset = dataset
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE,random_bg_folder_path=cfg.DIR.RANDOM_BG_PATH),#
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
    ])
    
    while True:
        train_loader = utils.datasetloader.dataloader(cfg,"train")
        sizeofdataset=len(train_loader)
        random.shuffle(train_loader)
        train_loader_permute = utils.datasetloader.dataloader_permutev2(train_loader)
        steps= int(sizeofdataset/cfg.CONST.BATCH_SIZE)
        for i in range(steps):
            x=train_loader_permute[0][i*cfg.CONST.BATCH_SIZE:(i+1)*cfg.CONST.BATCH_SIZE]
            y=train_loader_permute[1][i*cfg.CONST.BATCH_SIZE:(i+1)*cfg.CONST.BATCH_SIZE]
            outputx=[]
            outputy=[]
            print(outputx)
            for j in range(cfg.CONST.BATCH_SIZE):
                loadx,loady=mapload_funv3(x[j],y[j],"train")
                #print(np.array(loadx).shape)
                outputx.append(loadx)
                outputy.append(loady)
            #print(np.array(outputx).shape)
            #print(np.array(outputy).shape)
            #input()
            yield np.array(outputx),np.array(outputy)
        
def generator_trainv3(cfg,datatype):
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        #utils.data_transforms.ToTensor(),
    ])
    # Set up data loader
    train_dataset_loader = utils_origin.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg) 
    if datatype=='train':
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
            utils_origin.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                        batch_size=cfg.CONST.BATCH_SIZE,
                                                        num_workers=cfg.TRAIN.NUM_WORKER,
                                                        pin_memory=True,
                                                        shuffle=True,
                                                        drop_last=True)
    elif datatype=='val':
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
            utils_origin.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                        batch_size=cfg.CONST.BATCH_SIZE,
                                                        num_workers=cfg.TRAIN.NUM_WORKER,
                                                        pin_memory=True,
                                                        shuffle=True,
                                                        drop_last=True)
    elif datatype=='test':
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
            utils_origin.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                        batch_size=cfg.CONST.BATCH_SIZE,
                                                        num_workers=cfg.TRAIN.NUM_WORKER,
                                                        pin_memory=True,
                                                        shuffle=True,
                                                        drop_last=True)
    #val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    #val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
    #    utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
    #                                              batch_size=1,
     #                                             num_workers=1,
     #                                             pin_memory=True,
      #                                            shuffle=False)
    yield len(train_data_loader)
    while(1):
        for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) in enumerate(train_data_loader):
            #print(batch_idx)
            #print(taxonomy_names)
            #print(sample_names)
            if(cfg.CONST.N_VIEWS_RENDERING == 1):
                rendering_images=rendering_images.squeeze(dim=1)
            #print(rendering_images.numpy().shape)
            #print(ground_truth_volumes.numpy().shape)
            yield rendering_images.numpy(),np.expand_dims(ground_truth_volumes.numpy(),axis=4)

def valdata_gen():
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        #utils.data_transforms.ToTensor(),
    ])
    val_dataset_loader = utils_origin.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils_origin.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  shuffle=False)
    #yield len(val_data_loader)
    for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) in enumerate(val_data_loader):
        if(cfg.CONST.N_VIEWS_RENDERING == 1):
            rendering_images=rendering_images.squeeze(dim=1)
        rendering_images=rendering_images.squeeze(dim=0)
        ground_truth_volumes=ground_truth_volumes.squeeze(dim=0)
        #print(rendering_images.numpy().shape)
        #print(batch_idx)
        #input()
        yield rendering_images.numpy(),np.expand_dims(ground_truth_volumes.numpy(),axis=3)
def testdata_gen():
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        #utils.data_transforms.ToTensor(),
    ])
    test_dataset_loader = utils_origin.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils_origin.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  shuffle=False)
    #yield len(val_data_loader)
    for batch_idx, (taxonomy_names, sample_names, rendering_images, ground_truth_volumes) in enumerate(test_data_loader):
        if(cfg.CONST.N_VIEWS_RENDERING == 1):
            rendering_images=rendering_images.squeeze(dim=1)
        rendering_images=rendering_images.squeeze(dim=0)
        ground_truth_volumes=ground_truth_volumes.squeeze(dim=0)
        #print(rendering_images.numpy().shape)
        #print(batch_idx)
        #input()
        yield rendering_images.numpy(),np.expand_dims(ground_truth_volumes.numpy(),axis=3)
def mapload_funv3(x,y,dataset_type):
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    if(dataset_type=='train'):
      transforms = utils.data_transforms.Compose([
          utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
          utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),#,random_bg_folder_path=cfg.DIR.RANDOM_BG_PATH
          utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
          utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
          utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
          utils.data_transforms.RandomFlip(),
          utils.data_transforms.RandomPermuteRGB(),
      ])
    else:
      transforms = utils.data_transforms.Compose([
          utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
          utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
          utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
      ])
    #rendering_images=[]
    inputs = np.array([])
    #print('a')
    for i in range(cfg.CONST.N_VIEWS_RENDERING):
        xi = x[i]
        rendering_image = cv2.imread(xi, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        rendering_image = cv2.resize(rendering_image, (cfg.CONST.IMG_H, cfg.CONST.IMG_W))
        #rendering_images.append(rendering_image)
        rendering_image = np.expand_dims(rendering_image, axis=0)
        #if len(inputs) == 0:
        if i == 0:
            inputs = rendering_image
        else:
            inputs = np.append(inputs,rendering_image,axis=0)
    #input()
    inputs=transforms(inputs)
    #y = y.decode()
    with open(y, 'rb') as f:
        volume = utils.binvox_rw.read_as_3d_array(f)
        volume = volume.data.astype(np.float32)
    if(cfg.CONST.N_VIEWS_RENDERING == 1):
        return inputs[0], volume
    else:
        return inputs, volume
   
def mapload_funv2(x,y,dataset_type):
    x=x.numpy()
    y=y.numpy()
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    if(dataset_type=='train'):
      transforms = utils.data_transforms.Compose([
          utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
          utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),#,random_bg_folder_path=cfg.DIR.RANDOM_BG_PATH
          utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
          utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
          utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
          utils.data_transforms.RandomFlip(),
          utils.data_transforms.RandomPermuteRGB(),
      ])
    else:
      transforms = utils.data_transforms.Compose([
          utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
          utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
          utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
      ])
    #rendering_images=[]
    inputs = np.array([])
    #print('a')
    for i in range(cfg.CONST.N_VIEWS_RENDERING):
        xi = x[i]
        #print(xi)
        rendering_image = cv2.imread(xi.decode(), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        rendering_image = cv2.resize(rendering_image, (cfg.CONST.IMG_H, cfg.CONST.IMG_W))
        #rendering_images.append(rendering_image)
        rendering_image = np.expand_dims(rendering_image, axis=0)
        #if len(inputs) == 0:
        if i == 0:
            inputs = rendering_image
        else:
            inputs = np.append(inputs,rendering_image,axis=0)
    #input()
    inputs=transforms(inputs)
    y = y.decode()
    with open(y, 'rb') as f:
        volume = utils.binvox_rw.read_as_3d_array(f)
        volume = volume.data.astype(np.float32)
    if(cfg.CONST.N_VIEWS_RENDERING == 1):
        return inputs[0], volume
    else:
        return inputs, volume


def generator(cfg, dataset,sizeofdataset):
    backupdataset= dataset
    epochdataset = dataset
    while True:
        epochdataset = backupdataset
        epochdataset.shuffle(sizeofdataset)
        epochdataset = epochdataset.map(lambda x, y: tf.py_function(utils.datasetloader.mapload_fun, inp=[x, y,'train'], Tout=[tf.float32, tf.float32]),num_parallel_calls=tf.data.AUTOTUNE)
        epochdataset = epochdataset.map(utils.preprocessing.preprocessing_train,num_parallel_calls=tf.data.AUTOTUNE)
        epochdataset = epochdataset.batch(cfg.CONST.BATCH_SIZE)
        for item in epochdataset:
            yield item


def arrange_and_shuffle(cfg, files):
    loader=[]
    #count
    count=0
    for item in files:
        images=item["rendering_images"]
        count = count + len(images)
    ampty_count=0
    while 1:
        for i in range(len(files)):
            if len(files[i]["rendering_images"])<cfg.CONST.N_VIEWS_RENDERING:
                ampty_count = ampty_count + 1
                continue
            else:
                img=[]
                for j in range(cfg.CONST.N_VIEWS_RENDERING):
                    img.append(files[i]["rendering_images"].pop(random.randint(0,len(files[i]["rendering_images"])-1)))
                loader.append([img,files[i]["volume"]])
        if ampty_count >= len(files):
            break

    return loader

def dataloader_permute(loader):
    x=[]
    y=[]
    for i in range(len(loader)):
        x.append(loader[i][0])
        y.append(loader[i][1])
    loader_permute = (tf.constant(x),tf.constant(y))
    return loader_permute
def dataloader_permutev2(loader):
    x=[]
    y=[]
    for i in range(len(loader)):
        x.append(loader[i][0])
        y.append(loader[i][1])
    loader_permute = (x,y)
    return loader_permute
def mapload_fun(x,y,dataset_type):
    x=x.numpy()
    y=y.numpy()
    if(cfg.CONST.N_VIEWS_RENDERING == 1):
        x=x[0].decode()
        y=y.decode()
        rendering_image = cv2.imread(x, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        rendering_image = cv2.resize(rendering_image, (cfg.CONST.IMG_H, cfg.CONST.IMG_W))
        #randombackground
        img_height, img_width, img_channels = cfg.CONST.IMG_H, cfg.CONST.IMG_W, rendering_image.shape[2]
        if img_channels == 4:
            if dataset_type == 'train':
                random_color = cfg.TRAIN.RANDOM_BG_COLOR_RANGE
            else:
                random_color = cfg.TEST.RANDOM_BG_COLOR_RANGE
            r, g, b = np.array([
                np.random.randint(random_color[i][0], random_color[i][1] + 1)
                for i in
                range(3)
            ]) / 255.
            random_bg = None
            alpha = (np.expand_dims(rendering_image[:, :, 3], axis=2) == 0).astype(np.float32)
            rendering_image = rendering_image[:, :, :3]
            bg_color = random_bg if random.randint(0, 1) and random_bg is not None else np.array([[[r, g, b]]])
            rendering_image = alpha * bg_color + (1 - alpha) * rendering_image
        else:
            pass
        with open(y, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)
        return rendering_image, volume
    #2.0
    else:
        #rendering_images=[]
        inputs = np.array([])
        for i in range(cfg.CONST.N_VIEWS_RENDERING):
            xi = x[i]
            rendering_image = cv2.imread(xi.decode(), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            rendering_image = cv2.resize(rendering_image, (cfg.CONST.IMG_H, cfg.CONST.IMG_W))
            # randombackground
            img_height, img_width, img_channels = cfg.CONST.IMG_H, cfg.CONST.IMG_W, rendering_image.shape[2]
            if img_channels == 4:
                if dataset_type == 'train':
                    random_color = cfg.TRAIN.RANDOM_BG_COLOR_RANGE
                else:
                    random_color = cfg.TEST.RANDOM_BG_COLOR_RANGE
                r, g, b = np.array([
                    np.random.randint(random_color[i][0], random_color[i][1] + 1)
                    for i in
                    range(3)
                ]) / 255.
                random_bg = None
                alpha = (np.expand_dims(rendering_image[:, :, 3], axis=2) == 0).astype(np.float32)
                rendering_image = rendering_image[:, :, :3]
                bg_color = random_bg if random.randint(0, 1) and random_bg is not None else np.array([[[r, g, b]]])
                rendering_image = alpha * bg_color + (1 - alpha) * rendering_image
            else:
                pass
            #rendering_images.append(rendering_image)
            rendering_image = np.expand_dims(rendering_image, axis=0)
            if len(inputs) == 0:
                inputs = rendering_image
            else:
                inputs = np.append(inputs,rendering_image,axis=0)
        y = y.decode()
        with open(y, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)
        return inputs, volume
    #2.0

def dataloader(cfg,dataset_type):
    with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        dataset_taxonomy = json.loads(file.read())
    files = []
    # Load data for each category
    for taxonomy in dataset_taxonomy:
        taxonomy_folder_name = taxonomy['taxonomy_id']
        print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' %
              (dt.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
        samples = []
        if dataset_type == "train":
            samples = taxonomy['train']
            #if(cfg.CONST.N_VIEWS_RENDERING==1):
                #samples=samples[:int(len(samples)/2)]
                #samples=samples[:int(len(samples)/3*2)]
            #else:
                #samples=samples[int(len(samples)/2):]
                #samples=samples[int(len(samples)/3*2):]
        elif dataset_type == "test":
            samples = taxonomy['test']
        elif dataset_type == "val":
            samples = taxonomy['val']
        files.extend(get_files_of_taxonomy(cfg,taxonomy_folder_name, samples,dataset_type))
    return arrange_and_shuffle(cfg, files)
'''
    dataset=[]
    for item in files:
        rendering_images, volume=item["rendering_images"],item["volume"]
        select_images=[
            rendering_images[i]
            for i in random.sample(range(len(rendering_images)), cfg.CONST.N_VIEWS_RENDERING)
        ]
        dataset.append([select_images,volume])

    return dataset

    print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
    return ShapeNetDataset(dataset_type, files, n_views_rendering, transforms)

'''
def get_files_of_taxonomy(cfg,taxonomy_folder_name, samples,dataset_type):
    files_of_taxonomy = []
    for sample_idx, sample_name in enumerate(samples):
        # Get file path of volumes
        volume_file_path = cfg.DATASETS.SHAPENET.VOXEL_PATH % (taxonomy_folder_name, sample_name)
        if not os.path.exists(volume_file_path):
            print('[WARN] %s Ignore sample %s/%s since volume file not exists.' %
                  (dt.now(), taxonomy_folder_name, sample_name))
            continue

            # Get file list of rendering images
        img_file_path = cfg.DATASETS.SHAPENET.RENDERING_PATH % (taxonomy_folder_name, sample_name, 0)
        img_folder = os.path.dirname(img_file_path)
        total_views = len(os.listdir(img_folder))
        rendering_image_indexes = range(total_views)
        rendering_images_file_path = []
        for image_idx in rendering_image_indexes:
            img_file_path = cfg.DATASETS.SHAPENET.RENDERING_PATH % (taxonomy_folder_name, sample_name, image_idx)
            if(dataset_type=='test'):
                if(image_idx>cfg.CONST.N_VIEWS_RENDERING-1):
                    break
            if not os.path.exists(img_file_path):
                continue
            rendering_images_file_path.append(img_file_path)
        rendering_images_file_path = random.sample(rendering_images_file_path,cfg.CONST.N_VIEWS_RENDERING)
        if len(rendering_images_file_path) == 0:
            print('[WARN] %s Ignore sample %s/%s since image files not exists.' %
                (dt.now(), taxonomy_folder_name, sample_name))
            continue

            # Append to the list of rendering images
        files_of_taxonomy.append({
            'taxonomy_name': taxonomy_folder_name,
            'sample_name': sample_name,
            'rendering_images': rendering_images_file_path,
            'volume': volume_file_path,
        })

            # Report the progress of reading dataset
            # if sample_idx % 500 == 499 or sample_idx == n_samples - 1:
            #     print('[INFO] %s Collecting %d of %d' % (dt.now(), sample_idx + 1, n_samples))

    return files_of_taxonomy

