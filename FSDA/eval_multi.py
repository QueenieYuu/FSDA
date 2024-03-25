
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

bceloss = torch.nn.BCELoss(reduction='none')
seed = 3377
savefig = False
get_hd = True
model_save = True
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model-file', type=str, default='./logs/train_source/best_model.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain1')
    parser.add_argument('--source', type=str, default='Domain3')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='Data/Fundus/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file_1 = './logs/train_target/best_model.pth.tar'
    # model_file_2 = './logs/train_target/best_model_2.pth.tar'
    # model_file_3 = './logs/train_target/best_model_3.pth.tar'

    # 1. dataset
    composed_transforms_train = transforms.Compose([
        tr.Resize(512),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train)
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)
    db_source = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_test)

    train_loader = DataLoader(db_train, batch_size=8, shuffle=False, num_workers=1)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    # source_loader = DataLoader(db_source, batch_size=1, shuffle=False, num_workers=1)

    # 2. model
    model_1 = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()
    # model_2 = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()
    # model_3 = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    model_eval_1 = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()
    # model_eval_2 = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()
    # model_eval_3 = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    print('==> Loading %s model file: %s' %
          (model_1.__class__.__name__, model_file_1))
    # print('==> Loading %s model file: %s' %
    #       (model_2.__class__.__name__, model_file_2))
    # print('==> Loading %s model file: %s' %
    #       (model_3.__class__.__name__, model_file_3))

    checkpoint_1 = torch.load(model_file_1)
    # checkpoint_2 = torch.load(model_file_2)
    # checkpoint_3 = torch.load(model_file_3)

    model_1.load_state_dict(checkpoint_1['model_state_dict'])
    # model_2.load_state_dict(checkpoint_2['model_state_dict'])
    # model_3.load_state_dict(checkpoint_3['model_state_dict'])

    model_1.train()
    # model_2.train()
    # model_3.train()

    best_val_cup_dice = 0.0;
    best_val_disc_dice = 0.0;
    best_avg = 0.0

    iter_num = 0

    #test
    model_eval_1.train()
    pretrained_dict = model_1.state_dict()
    model_dict = model_eval_1.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_eval_1.load_state_dict(pretrained_dict)

    # model_eval_2.train()
    # pretrained_dict = model_2.state_dict()
    # model_dict = model_eval_2.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_eval_2.load_state_dict(pretrained_dict)
    #
    # model_eval_3.train()
    # pretrained_dict = model_3.state_dict()
    # model_dict = model_eval_3.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_eval_3.load_state_dict(pretrained_dict)

    val_cup_dice = 0.0;val_disc_dice = 0.0;datanum_cnt = 0.0
    cup_hd = 0.0; disc_hd = 0.0;datanum_cnt_cup = 0.0;datanum_cnt_disc = 0.0
    with torch.no_grad():
        for batch_idx, (sample) in enumerate(test_loader):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            prediction_1, boundary_1, _= model_eval_1(data)
            # prediction_2, boundary_2, _= model_eval_2(data)
            # prediction_3, boundary_3, _ = model_eval_3(data)
            #
            # a, b = 0.3333, 0.3333
            # prediction = a*prediction_1 + b*prediction_2 + (1.0-a-b)*prediction_3

            prediction = torch.sigmoid(prediction_1)

            # draw_mask(prediction.data.cpu()[0].numpy(), os.path.join('./results', 'fundus'), str(img_name))
            mask_cup, mask_disc, mask = colorize_mask(prediction)
            mask_cup.save('./results/fundus/cup/%s' % (str(sample['img_name'][0])))
            mask_disc.save('./results/fundus/disc/%s' % (str(sample['img_name'][0])))
            mask.save('./results/fundus/mask/%s' % (str(sample['img_name'][0])))

            target_numpy = target.data.cpu()
            prediction = prediction.data.cpu()
            prediction[prediction>0.75] = 1;prediction[prediction <= 0.75] = 0

            cup_dice = dice_coefficient_numpy(prediction[:,0, ...], target_numpy[:, 0, ...])
            disc_dice = dice_coefficient_numpy(prediction[:,1, ...], target_numpy[:, 1, ...])

            for i in range(prediction.shape[0]):
                hd_tmp = hd_numpy(prediction[i, 0, ...], target_numpy[i, 0, ...], get_hd)
                if np.isnan(hd_tmp):
                    datanum_cnt_cup -= 1.0
                else:
                    cup_hd += hd_tmp

                hd_tmp = hd_numpy(prediction[i, 1, ...], target_numpy[i, 1, ...], get_hd)
                if np.isnan(hd_tmp):
                    datanum_cnt_disc -= 1.0
                else:
                    disc_hd += hd_tmp

            val_cup_dice += np.sum(cup_dice)
            val_disc_dice += np.sum(disc_dice)

            datanum_cnt += float(prediction.shape[0])
            datanum_cnt_cup += float(prediction.shape[0])
            datanum_cnt_disc += float(prediction.shape[0])

        val_cup_dice /= datanum_cnt
        val_disc_dice /= datanum_cnt
        cup_hd /= datanum_cnt_cup
        disc_hd /= datanum_cnt_disc
        if (val_cup_dice+val_disc_dice)/2.0>best_avg:
            best_val_cup_dice = val_cup_dice; best_val_disc_dice = val_disc_dice; best_avg = (val_cup_dice+val_disc_dice)/2.0
            best_cup_hd = cup_hd; best_disc_hd = disc_hd; best_avg_hd = (best_cup_hd+best_disc_hd)/2.0

        # if not os.path.exists('./logs/train_target'):
        #     os.mkdir('./logs/train_target')
        # if args.dataset == 'Domain1':
        #     savefile = './logs/train_target/' + 'D1_' + 'checkpoint_%d.pth.tar' % epoch_num
        # elif args.dataset == 'Domain2':
        #     savefile = './logs/train_target/' + 'D2_' + 'checkpoint_%d.pth.tar' % epoch_num
        # if model_save:
        #     torch.save({
        #         'model_state_dict': model_eval_1.state_dict(),
        #         'best_mean_dice': best_avg,
        #         'best_cup_dice': best_val_cup_dice,
        #         'best_disc_dice': best_val_disc_dice,
        #     }, savefile)

        print("cup: %.4f disc: %.4f avg: %.4f cup: %.4f disc: %.4f avg: %.4f" %
              (val_cup_dice, val_disc_dice, (val_cup_dice+val_disc_dice)/2.0, cup_hd, disc_hd, (cup_hd+disc_hd)/2.0))
        print("best cup: %.4f best disc: %.4f best avg: %.4f best cup: %.4f best disc: %.4f best avg: %.4f" %
              (best_val_cup_dice, best_val_disc_dice, best_avg, best_cup_hd, best_disc_hd, best_avg_hd))
        #model.train()
