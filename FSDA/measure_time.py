#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval
import cv2
import torch.backends.cudnn as cudnn
import random
from tensorboardX import SummaryWriter

from PIL import Image



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='./logs/train_source/best_model.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain2')
    parser.add_argument('--source', type=str, default='Domain3')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='Data/Fundus/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    model_file = './logs/train_source/best_model.pth.tar'
    model = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    # device = torch.device('cuda')
    # model.to(device)

    image = Image.open('gdrishtiGS_004.png')

    transform = transforms.Compose([transforms.PILToTensor()])
    image_tensor = transform(image).cuda().float()
    image_tensor = image_tensor[None, :]

    dummy_input = torch.randn(1, 3,224,224,dtype=torch.float).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
       _ = model(image_tensor)
    # MEASURE PERFORMANCE
    with torch.no_grad():
      for rep in range(repetitions):
         starter.record()
         _ = model(image_tensor)
         ender.record()
         # WAIT FOR GPU SYNC
         torch.cuda.synchronize()
         curr_time = starter.elapsed_time(ender)
         timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)
