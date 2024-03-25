
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

from torch.nn import functional as F
from networks.models import Scalar

from utils.losses import EntKLProp

bceloss = torch.nn.BCELoss(reduction='none')

def entloss(p):
    # y1 = -1.0*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C))
    n, c, h, w = p.size()
    y1 = -torch.mul(p, torch.log2(p + 1e-30)) / np.log2(c)
    ent = torch.mean(y1)
    return ent

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=0)
    target_softmax = F.softmax(target_logits, dim=0)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    if global_step <= args.iter_pretrain:
        alpha = 0
    else:
        alpha = min(1 - 1 / (global_step - args.iter_pretrain + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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
    # parser.add_argument('--model-file', type=str, default='./logs/train_source/best_model_d3.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain3')
    # parser.add_argument('--source', type=str, default='Domain3')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='Data/Fundus/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument('--ema-decay', type=float,default=3)
    parser.add_argument('--iter-pretrain', type=int, default=100, help='pretraining iteration for student network')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file_d1 = './logs/train_source/best_model_d1.pth.tar'
    model_file_d2 = './logs/train_source/best_model_d2.pth.tar'
    model_file_d3 = './logs/train_source/best_model_d3.pth.tar'
    model_file_d4 = './logs/train_source/best_model_d4.pth.tar'

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

    train_loader = DataLoader(db_train, batch_size=8, shuffle=False, num_workers=1,drop_last=True)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1,drop_last=True)
    # fda_loader = DataLoader(db_fda, batch_size=2, shuffle=False, num_workers=1,drop_last=True)

    # 2. model
    model_t1 = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_t2 = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_student = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_eval = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    if torch.cuda.is_available():
        model_t1 = model_t1.cuda()
    checkpoint = torch.load(model_file_d1)
    model_t1.load_state_dict(checkpoint['model_state_dict'])
    model_t1.train()

    if torch.cuda.is_available():
        model_t2 = model_t2.cuda()
    checkpoint = torch.load(model_file_d2)
    model_t2.load_state_dict(checkpoint['model_state_dict'])
    model_t2.train()

    if torch.cuda.is_available():
        model_student = model_student.cuda()
    checkpoint = torch.load(model_file_d4)
    model_student.load_state_dict(checkpoint['model_state_dict'])
    model_student.train()

    w = 2*torch.rand((2))-1
    netG_list = [Scalar(w[i]).cuda() for i in range(2)]


    if args.dataset=="Domain1":
        npfilename = './results/prototype/pseudolabel_D1.npz'
    elif args.dataset=="Domain2":
        npfilename = './results/prototype/pseudolabel_D2.npz'
    elif args.dataset=="Domain3":
        npfilename = './results/prototype/pseudolabel_D3.npz'
    elif args.dataset=="Domain4":
        npfilename = './results/prototype/pseudolabel_D4.npz'

    npdata = np.load(npfilename, allow_pickle=True)
    pseudo_label_dic = npdata['arr_0'].item()

    # var_list = model.named_parameters()

    optim_gen = torch.optim.Adam(model_student.parameters(), lr=0.002, betas=(0.9, 0.99))
    best_val_cup_dice = 0.0;
    best_val_disc_dice = 0.0;
    best_avg = 0.0

    loss_fn = EntKLProp()

    iter_num = 0
    for epoch_num in tqdm.tqdm(range(100), ncols=70):

        model_student.train()
        # #torch.cuda.empty_cache()
        #
        for batch_idx, (sample) in enumerate(train_loader):

            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            for param in model_student.parameters():
                param.requires_grad = True
            optim_gen.zero_grad()

            pred_stu, _, _, _ = model_student(data)

            outputs_all = torch.zeros(2, pred_stu.shape[0], 2,  pred_stu.shape[2],  pred_stu.shape[3])
            weights_all = torch.ones(pred_stu.shape[0],  2)
            outputs_all_w = torch.zeros(pred_stu.shape[0], 2, pred_stu.shape[2], pred_stu.shape[3])

            pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
            pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
            loss_seg = bceloss(torch.sigmoid(pred_stu), pseudo_label)
            # loss_seg.mean().backward()

            #ADAMI
            # loss_1, loss_cons_prior,est_prop =  loss_fn(pred_stu, pseudo_label)
            # loss = loss_1 + loss_cons_prior + loss_seg.mean()
            #ADAMI

            noise = torch.clamp(torch.randn_like(data) * 0.1, -0.2, 0.2)
            data_noise = data + noise

            pred_t1, feature_1, _, _ = model_t1(data_noise)
            pred_t2, feature_2, _, _ = model_t2(data_noise)

            # DECISION
            # weights_test_1 = netG_list[0](feature_1)
            # weights_test_2 = netG_list[1](feature_2)
            # weights_all[:,0] = weights_test_1.squeeze()
            # weights_all[:,1] = weights_test_2.squeeze()
            # DECISION

            loss_consis_1 = softmax_mse_loss(pred_stu, pred_t1).mean()
            loss_consis_2 = softmax_mse_loss(pred_stu, pred_t2).mean()

            # SHOT
            # softmax_out = F.softmax(pred_t1, dim=0)
            # entropy_loss_1 = torch.mean(entloss(softmax_out))
            #
            # msoftmax = softmax_out.mean(dim=1)
            # entropy_loss_2 = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5)) * 0.00001
            #
            # softmax_out = F.softmax(pred_t2, dim=0)
            # entropy_loss_1 += torch.mean(entloss(softmax_out))
            #
            # msoftmax = softmax_out.mean(dim=1)
            # entropy_loss_2 += torch.sum(-msoftmax * torch.log(msoftmax + 1e-5)) * 0.00001
            #
            # entropy_loss =  entropy_loss_1 - entropy_loss_2
            # SHOT

            w_diff = torch.tensor(0.).cuda()
            for w, w_t in zip(model_student.parameters(), model_t1.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            com_loss_1 = 1e-1 / 2. * w_diff

            w_diff = torch.tensor(0.).cuda()
            for w, w_t in zip(model_student.parameters(), model_t2.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            com_loss_2 = 1e-1 / 2. * w_diff

            outputs_all[0] = pred_t1
            outputs_all[1] = pred_t2

            weights_all[:,0] = loss_consis_1
            weights_all[:,1] = loss_consis_2

            outputs_all = torch.transpose(outputs_all, 0, 1)

            for i in range(data.shape[0]):
                # print(torch.transpose(torch.transpose(outputs_all[i],0,3),0,1).shape)
                # print(torch.mean(torch.mean(torch.transpose(torch.transpose(weights_all[i],0,2),1,2), 1), 1).shape)
                outputs_all_w[i] = torch.matmul(torch.transpose(torch.transpose(outputs_all[i],0,3),0,1),weights_all[i])
            outputs_all_w = outputs_all_w.cuda()

            softmax_out = F.softmax(outputs_all_w, dim=0)
            entropy_loss_1= torch.mean(entloss(softmax_out))

            msoftmax = softmax_out.mean(dim=1)
            entropy_loss_2 = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5)) * 0.00001

            entropy_loss =  entropy_loss_1 - entropy_loss_2

            loss = loss_seg.mean() + entropy_loss + torch.mean(loss_consis_1 + loss_consis_2) + torch.mean(com_loss_1 + com_loss_2)
            loss.backward()

            optim_gen.step()
            iter_num = iter_num + 1

        #test
        model_eval.train()
        pretrained_dict = model_student.state_dict()
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

                mask_cup, mask_disc, mask = colorize_mask(prediction)
                # mask_cup.save('./results/fundus/cup/%s' % (str(sample['img_name'][0])))
                # mask_disc.save('./results/fundus/disc/%s' % (str(sample['img_name'][0])))
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

        if not os.path.exists('./logs/train_target'):
            os.mkdir('./logs/train_target')
        if args.dataset == 'Domain1':
            savefile = './logs/train_target/' + 'D1_' + 'checkpoint_%d.pth.tar' % epoch_num
        elif args.dataset == 'Domain2':
            savefile = './logs/train_target/' + 'D2_' + 'checkpoint_%d.pth.tar' % epoch_num
        elif args.dataset == 'Domain3':
            savefile = './logs/train_target/' + 'D3_' + 'checkpoint_%d.pth.tar' % epoch_num
        elif args.dataset == 'Domain4':
            savefile = './logs/train_target/' + 'D4_' + 'checkpoint_%d.pth.tar' % epoch_num
        if model_save:
            torch.save({
                'model_state_dict': model_student.state_dict(),
                'best_mean_dice': best_avg,
                'best_cup_dice': best_val_cup_dice,
                'best_disc_dice': best_val_disc_dice,
            }, savefile)

        print("cup: %.4f disc: %.4f avg: %.4f cup: %.4f disc: %.4f avg: %.4f" %
              (val_cup_dice, val_disc_dice, (val_cup_dice+val_disc_dice)/2.0, cup_hd, disc_hd, (cup_hd+disc_hd)/2.0))
        print("best cup: %.4f best disc: %.4f best avg: %.4f best cup: %.4f best disc: %.4f best avg: %.4f" %
              (best_val_cup_dice, best_val_disc_dice, best_avg, best_cup_hd, best_disc_hd, best_avg_hd))
        model_student.train()
