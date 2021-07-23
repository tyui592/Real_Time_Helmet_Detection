import os
import time
import argparse

import torch
import random
import numpy

from utils import save_pickle, load_pickle

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-no', type=int, nargs='+',
            help='Number of GPU ID to use, 0~N: GPU, -1: CPU', default=[0])
    parser.add_argument('--random-seed', type=int,
            help='random seed for reproducible experiments', default=777)

    # train
    parser.add_argument('--train-flag', action='store_true',
            help='set this flag for training', default=False)
    parser.add_argument('--data', type=str,
            help='data path for training or evaluation', default=None)
    parser.add_argument('--batch-size', type=int,
            help='batch size', default=16)
    parser.add_argument('--sub-divisions', type=int,
            help='optimize every N iterations for gradient accumulation', default=1)
    parser.add_argument('--start-epoch', type=int,
            help='start epoch', default=0)
    parser.add_argument('--end-epoch', type=int,
            help='end epoch', default=100)
    parser.add_argument('--num-workers', type=int,
            help='number of workers for data loading', default=8)

    # amp
    parser.add_argument('--amp', action='store_true',
            help='automatic mixed precision flag', default=False)

    # ddp(distributed data parallel)
    parser.add_argument('--world-size', type=int,
            help='for distributed data parallel', default=1)
    parser.add_argument('--rank', type=int,
            help='for distributed data parallel', default=0)
    parser.add_argument('--dist-backend', type=str,
            help='for distributed data parallel', default='nccl')
    parser.add_argument('--dist-url', type=str,
            help='for distributed data parallel', default='tcp://localhost:29500')

    # evaluation and demo
    parser.add_argument('--imsize', type=int,
            help='when evaluation or demo run, the image resized by imsize x imsize', default=None)
    parser.add_argument('--topk', type=int,
            help='extract topk peak predictions', default=100)
    parser.add_argument('--conf-th', type=float,
            help='confidence threshold', default=0.0)
    parser.add_argument('--nms-th', type=float,
            help='nms threshold', default=0.5)
    parser.add_argument('--pool-size', type=int,
            help='pool(it is used to find peak value in heatmap) size', default=3)
    parser.add_argument('--model-load', type=str,
            help='check_point path', default=None)
    parser.add_argument('--nms', type=str,
            help='select nms algorithm (nms | soft-nms)', default='nms')
    parser.add_argument('--fontsize', type=int,
            help='fontsize for demo, 0: dont write score and class in the image', default=10)

    # augmentation
    # reference: https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html#a-simple-example
    parser.add_argument('--crop-percent', type=float, nargs='+',
            help='range(min, max), how many crop the image', default=[0.0, 0.1])
    parser.add_argument('--color-multiply', type=float, nargs='+',
            help='range(min, max), how many adjust the brightness', default=[1.2, 1.5])
    parser.add_argument('--translate-percent', type=float,
            help='ratio, how many translate the image', default=0.1)
    parser.add_argument('--affine-scale', type=float, nargs='+',
            help='range(min, ratio), how many scaling the image', default=[0.5, 1.5])
    parser.add_argument('--multiscale_flag', action='store_true',
            help='training with multi-resolution images the resolution is randomly selected per every iteration', default=False)
    parser.add_argument('--multiscale', type=int, nargs='+',
            help='[min, max, step] if multiscale_flag set False network train with the max size', default=[320, 512, 64])

    # loss
    parser.add_argument('--hm-weight', type=float,
            help='heat map loss weight', default=1.0)
    parser.add_argument('--offset-weight', type=float,
            help='offset loss weight', default=1.0)
    parser.add_argument('--size-weight', type=float,
            help='size(wh) loss weight', default=0.1)
    parser.add_argument('--focal-alpha', type=float,
            help='alpha for focal loss(heatmap)', default=2.0)
    parser.add_argument('--focal-beta', type=float,
            help='beta for focal loss(heatmap)', default=4.0)

    # network
    parser.add_argument('--scale_factor', type=int,
            help='downsampling scale from image to heatmap', default=4)
    parser.add_argument('--num-cls', type=int,
            help='number of classes', default=2)
    parser.add_argument('--pretrained', type=str,
            help='select pretrained backbone (scratch | imagenet)', default='imagenet')
    parser.add_argument('--normalized-coord', action='store_true',
            help='predict normalized(relative) offset ans size of bounding box', default=False)
    ## backbone - hourglass
    parser.add_argument('--num-stack', type=int,
            help='number of stack in hourglass network', default=1)
    parser.add_argument('--hourglass-inch', type=int,
            help='number of channels for hougrglass networks', default=128)
    parser.add_argument('--increase-ch', type=int,
            help='in the hougralss network, more deep layer has more channels by this factor', default=0)
    parser.add_argument('--activation', type=str,
            help='activation funciton', default='ReLU')
    parser.add_argument('--pool', type=str,
            help='pooling function', default='Max')
    ## neck
    parser.add_argument('--neck-activation', type=str,
            help='activation funciton', default='ReLU')
    parser.add_argument('--neck-pool', type=str,
            help='pooling function (None | SPP)', default='None')

    # optimization
    parser.add_argument('--lr', type=float,
            help='learning rate, select lr more carefully when use amp', default=5e-4)
    parser.add_argument('--optim', type=str,
            help='optimization algorithm', default='Adam')
    parser.add_argument('--lr-milestone', type=int, nargs='+',
            help='epoch for adjust lr', default=[50, 90])
    parser.add_argument('--lr-gamma', type=float,
            help='scale factor', default=0.1)

    # log
    parser.add_argument('--print-interval', type=int,
            help='print logs every N iterations', default=100)
    parser.add_argument('--save-path', type=str,
            help='path to save results', default='./WEIGHTS/')

    return parser.parse_args()


def get_arguments():
    args = build_parser()
    # set random seed for reproducible experiments
    # reference: https://github.com/pytorch/pytorch/issues/7068
    random.seed(args.random_seed)
    numpy.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # these flags can affect performance, selec carefully
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    os.makedirs(args.save_path, exist_ok=True)
    if args.train_flag:
        os.makedirs(os.path.join(args.save_path, 'training_log'), exist_ok=True)
    else:
        loaded_args = load_pickle(os.path.join(os.path.dirname(args.model_load), 'argument.pickle'))
        args = update_arguments_for_eval(args, loaded_args)

    # cuda setting
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(map(str, args.gpu_no))

    with open(os.path.join(args.save_path, 'argument.txt'), 'w') as f:
        for key, value in sorted(vars(args).items()):
            f.write('%s: %s'%(key, value) + '\n')

    save_pickle(os.path.join(args.save_path, 'argument.pickle'), args)
    return args

def update_arguments_for_eval(old, new):
    targets = ['scale_factor', 'num_cls', 'pretrained', 'normalized_coord',
               'num_stack', 'hourglass_inch', 'increase_ch', 'activation', 'pool',
               'neck_activation', 'neck_pool']

    for target in targets:
        old.__dict__[target] = new.__dict__[target]

    return old
