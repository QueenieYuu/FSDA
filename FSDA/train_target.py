
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
    # db_fda = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train, set = 'fda')

    train_loader = DataLoader(db_train, batch_size=2, shuffle=False, num_workers=1,drop_last=True)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1,drop_last=True)
    # fda_loader = DataLoader(db_fda, batch_size=2, shuffle=False, num_workers=1,drop_last=True)

    # 2. model
    model = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_eval = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.train()

    if args.dataset=="Domain2":
        npfilename = './results/prototype/pseudolabel_D2.npz'
        npfilename_fda = './results/prototype/pseudolabel_D2_fda.npz'
        # npfilename_2 = './results/prototype/pseudolabel_D2_0.85.npz'
        # npfilename_3 = './results/prototype/pseudolabel_D2_0.65.npz'
    elif args.dataset=="Domain1":
        npfilename = './results/prototype/pseudolabel_D1.npz'
        npfilename_fda = './results/prototype/pseudolabel_D1_fda.npz'


    npdata = np.load(npfilename, allow_pickle=True)
    # npdata_2 = np.load(npfilename_2, allow_pickle=True)
    # npdata_3 = np.load(npfilename_3, allow_pickle=True)
    pseudo_label_dic = npdata['arr_0'].item()
    # pseudo_label_dic_2 = npdata_2['arr_0'].item()
    # pseudo_label_dic_3 = npdata_3['arr_0'].item()
    # uncertain_dic = npdata['arr_1'].item()
    # proto_pseudo_dic = npdata['arr_2'].item()

    npdata_fda = np.load(npfilename_fda, allow_pickle=True)
    pseudo_label_dic_fda = npdata_fda['arr_0'].item()


    var_list = model.named_parameters()

    optim_gen = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99))
    best_val_cup_dice = 0.0;
    best_val_disc_dice = 0.0;
    best_avg = 0.0

    iter_num = 0
    for epoch_num in tqdm.tqdm(range(30), ncols=70):
        # model.train()
        #
        # #torch.cuda.empty_cache()
        #
        # for batch_idx, (sample) in enumerate(train_loader):
        #     data, target, img_name = sample['image'], sample['map'], sample['img_name']
        #     if torch.cuda.is_available():
        #         data, target = data.cuda(), target.cuda()
        #     data, target = Variable(data), Variable(target)
        #
        #     try:
        #         id_, sample_fda = next(domain_fda_loader)
        #     except:
        #         domain_fda_loader = enumerate(fda_loader)
        #         id_, sample_fda = next(domain_fda_loader)
        #     data_fda, target_fda, img_name_fda = sample_fda['image'], sample_fda['map'], sample_fda['img_name']
        #     if torch.cuda.is_available():
        #         data_fda, target_fda = data_fda.cuda(), target_fda.cuda()
        #     data_fda, target_fda = Variable(data_fda), Variable(target_fda)
        #
        #     for param in model.parameters():
        #         param.requires_grad = True
        #     optim_gen.zero_grad()
        #
        #     prediction, _, _, _ = model(data)
        #     prediction_fda, _, _, _ = model(data_fda)
        #
        #     pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
        #     pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
        #
        #     pseudo_label_fda = [pseudo_label_dic_fda.get(key) for key in img_name_fda]
        #     pseudo_label_fda = torch.from_numpy(np.asarray(pseudo_label_fda)).float().cuda()
        # #
        # #     # pseudo_label_2 = [pseudo_label_dic_2.get(key) for key in img_name]
        # #     # pseudo_label_2 = torch.from_numpy(np.asarray(pseudo_label_2)).float().cuda()
        # #     #
        # #     # pseudo_label_3 = [pseudo_label_dic_3.get(key) for key in img_name]
        # #     # pseudo_label_3 = torch.from_numpy(np.asarray(pseudo_label_3)).float().cuda()
        # #     #
        # #     # a, b = 0.3333, 0.3333
        # #     # pseudo_label = a*pseudo_label_1 + b*pseudo_label_2 + (1.0-a-b)*pseudo_label_3
        # #     #
        #     # try:
        #     #     id_, sampleS = next(domain_s_loader)
        #     # except:
        #     #     domain_s_loader = enumerate(source_loader)
        #     #     id_, sampleS = next(domain_s_loader)
        #     # imageS, source_map, imageS_name = sampleS['image'], sampleS['map'], sampleS['img_name']
        #     #
        #     # if torch.cuda.is_available():
        #     #     imageS, source_map = imageS.cuda(), source_map.cuda()
        #     # imageS, source_map = Variable(imageS), Variable(source_map)
        #     #
        #     # oS, _, _, _ = model(imageS)
        # #     #
        # #     # loss_seg_S = bceloss(torch.sigmoid(oS), source_map)
        #     loss_seg_T = bceloss(torch.sigmoid(prediction), pseudo_label)
        #     loss_seg_T_fda = bceloss(torch.sigmoid(prediction_fda), pseudo_label_fda)
        # #     # loss_seg_T1 = bceloss(torch.sigmoid(prediction), pseudo_label_1)
        # #     # loss_seg_T2 = bceloss(torch.sigmoid(prediction), pseudo_label_2)
        # #     # loss_seg_T3 = bceloss(torch.sigmoid(prediction), pseudo_label_3)
        # #     #
        # #     # a, b = 0.3333, 0.3333
        # #     # loss_seg_T = a*loss_seg_T1 + b*loss_seg_T2 + (1.0-a-b)*loss_seg_T3
        # #
        #     loss_seg = loss_seg_T + loss_seg_T_fda
        #     loss_seg.mean().backward()
        #     optim_gen.step()
        #     iter_num = iter_num + 1

        #test
        model_eval.train()
        pretrained_dict = model.state_dict()
        model_dict = model_eval.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_eval.load_state_dict(pretrained_dict)

        val_cup_dice = 0.0;val_disc_dice = 0.0;datanum_cnt = 0.0
        cup_hd = 0.0; disc_hd = 0.0;datanum_cnt_cup = 0.0;datanum_cnt_disc = 0.0
        with torch.no_grad():
            for batch_idx, (sample) in enumerate(test_loader):
                data, target, img_name = sample['image'], sample['map'], sample['img_name']
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                prediction, boundary, _ = model_eval(data)
                prediction = torch.sigmoid(prediction)

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

        if not os.path.exists('./logs/train_target'):
            os.mkdir('./logs/train_target')
        if args.dataset == 'Domain1':
            savefile = './logs/train_target/' + 'D1_' + 'checkpoint_%d.pth.tar' % epoch_num
        elif args.dataset == 'Domain2':
            savefile = './logs/train_target/' + 'D2_' + 'checkpoint_%d.pth.tar' % epoch_num
        if model_save:
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_mean_dice': best_avg,
                'best_cup_dice': best_val_cup_dice,
                'best_disc_dice': best_val_disc_dice,
            }, savefile)

        print("cup: %.4f disc: %.4f avg: %.4f cup: %.4f disc: %.4f avg: %.4f" %
              (val_cup_dice, val_disc_dice, (val_cup_dice+val_disc_dice)/2.0, cup_hd, disc_hd, (cup_hd+disc_hd)/2.0))
        print("best cup: %.4f best disc: %.4f best avg: %.4f best cup: %.4f best disc: %.4f best avg: %.4f" %
              (best_val_cup_dice, best_val_disc_dice, best_avg, best_cup_hd, best_disc_hd, best_avg_hd))
        model.train()
