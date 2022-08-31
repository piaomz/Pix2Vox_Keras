# Developed by Mingzhe Piao 

import os
import sys
from argparse import ArgumentParser
from pprint import pprint

from config import cfg
#from core.train import train_net
#print("123")
from core.test import test_net
#print("223")
from core.predict import predict_net
#print("323")
from core.converttotfjs import convert_net
#print("423")
import numpy as np


#input()

#tf.test.is_gpu_available()

def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--predict', dest='predict', help='Test neural networks', action='store_true')
    parser.add_argument('--convert', dest='convert', help='Test neural networks', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHES = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(cfg.CONST.DEVICE))
    # Print config
    print('Use config:')
    pprint(cfg)
    if int(cfg.CONST.DEVICE)>=0:
      os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
      os.environ["CUDA_VISIBLE_DEVICES"] = str(int(cfg.CONST.DEVICE))
      import tensorflow as tf
      gpus = tf.config.experimental.list_physical_devices('GPU')
      print(gpus)
      if gpus:
          try:
              tf.config.experimental.set_virtual_device_configuration(gpus[int(0)], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=cfg.CONST.GPU_MEM)])
          except RuntimeError as e:
              print(e)
    
    # Start train/test process
    if args.test:
        # if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
        # test_net(cfg)
        test_net(cfg)
        # else:
        #    print('[FATAL] %s Please specify the file path of checkpoint.' % (dt.now()))
        #    sys.exit(2)
    elif args.predict:
        predict_net(cfg)
    elif args.convert:
        convert_net(cfg)
    else:
        from core.train import train_net
        train_net(cfg)


main()
