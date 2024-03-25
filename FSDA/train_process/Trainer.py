from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import time

import numpy as np
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from tensorboardX import SummaryWriter

import tqdm
import socket
from utils.metrics import *
from utils.Utils import *

from perturbations.augmentations import FDA_source_to_target
import random

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return torch.nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

class Trainer(object):

    def __init__(self, cuda, model_gen, model_dis, model_dis2, optimizer_gen, optimizer_dis, optimizer_uncertainty_dis,
                 val_loader, domain_loaderS_1, domain_loaderS_2, domain_loaderT, out, max_epoch, stop_epoch=None,
                 lr_gen=1e-3, lr_dis=1e-3, lr_decrease_rate=0.1, interval_validate=None, batch_size=8, warmup_epoch=10):
        self.cuda = cuda
        self.warmup_epoch = warmup_epoch
        self.model_gen = model_gen
        self.model_dis2 = model_dis2
        self.model_dis = model_dis
        self.optim_gen = optimizer_gen
        self.optim_dis = optimizer_dis
        self.optim_dis2 = optimizer_uncertainty_dis
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.domain_loaderS = domain_loaderS_1
        self.domain_loaderS_2 = domain_loaderS_2
        # self.domain_loaderS_3 = domain_loaderS_3
        self.domain_loaderT = domain_loaderT
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/cup_dice',
            'train/disc_dice',
            'train/loss_adv',
            'train/loss_D_same',
            'train/loss_D_diff',
            'valid/loss_CE',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.best_epoch = -1


    def validate(self):
        training = self.model_gen.training
        self.model_gen.eval()

        val_loss = 0.0
        val_cup_dice = 0.0
        val_disc_dice = 0.0
        datanum_cnt = 0.0
        metrics = []

        cup_hd = 0.0; disc_hd = 0.0;datanum_cnt_cup = 0.0;datanum_cnt_disc = 0.0

        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['image']
                target_map = sample['map']
                target_boundary = sample['boundary']
                if self.cuda:
                    data, target_map, target_boundary = data.cuda(), target_map.cuda(), target_boundary.cuda()
                with torch.no_grad():
                    predictions, boundary, _,_ = self.model_gen(data)

                loss = F.binary_cross_entropy_with_logits(predictions, target_map)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
                val_cup_dice += np.sum(dice_cup)
                val_disc_dice += np.sum(dice_disc)
                datanum_cnt += float(dice_cup.shape[0])

                target_numpy = target_map.data.cpu()
                prediction = torch.sigmoid(predictions)
                prediction = prediction.data.cpu()
                prediction[prediction>0.75] = 1;prediction[prediction <= 0.75] = 0

                for i in range(prediction.shape[0]):
                    hd_tmp = hd_numpy(prediction[i, 0, ...], target_numpy[i, 0, ...], True)
                    if np.isnan(hd_tmp):
                        datanum_cnt_cup -= 1.0
                    else:
                        cup_hd += hd_tmp

                    hd_tmp = hd_numpy(prediction[i, 1, ...], target_numpy[i, 1, ...], True)
                    if np.isnan(hd_tmp):
                        datanum_cnt_disc -= 1.0
                    else:
                        disc_hd += hd_tmp

                datanum_cnt_cup += float(prediction.shape[0])
                datanum_cnt_disc += float(prediction.shape[0])

            val_loss /= datanum_cnt
            val_cup_dice /= datanum_cnt
            val_disc_dice /= datanum_cnt

            cup_hd /= datanum_cnt_cup
            disc_hd /= datanum_cnt_disc

            metrics.append((val_loss, val_cup_dice, val_disc_dice, cup_hd, disc_hd))
            self.writer.add_scalar('val_data/loss_CE', val_loss, self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.val_loader)))

            self.writer.add_scalar('val_data/val_CUP_ASD', val_disc_dice, self.epoch * (len(self.val_loader)))
            self.writer.add_scalar('val_data/val_DISC_ASD', val_disc_dice, self.epoch * (len(self.val_loader)))

            mean_dice = val_cup_dice + val_disc_dice
            is_best = mean_dice > self.best_mean_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model_gen.__class__.__name__,
                    'optim_state_dict': self.optim_gen.state_dict(),
                    'optim_dis_state_dict': self.optim_dis.state_dict(),
                    'optim_dis2_state_dict': self.optim_dis2.state_dict(),
                    'model_state_dict': self.model_gen.state_dict(),
                    'model_dis_state_dict': self.model_dis.state_dict(),
                    'model_dis2_state_dict': self.model_dis2.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_gen),
                    'learning_rate_dis': get_lr(self.optim_dis),
                    'learning_rate_dis2': get_lr(self.optim_dis2),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.best_epoch))
            else:
                if (self.epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model_gen.__class__.__name__,
                    'optim_state_dict': self.optim_gen.state_dict(),
                    'optim_dis_state_dict': self.optim_dis.state_dict(),
                    'optim_dis2_state_dict': self.optim_dis2.state_dict(),
                    'model_state_dict': self.model_gen.state_dict(),
                    'model_dis_state_dict': self.model_dis.state_dict(),
                    'model_dis2_state_dict': self.model_dis2.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_gen),
                    'learning_rate_dis': get_lr(self.optim_dis),
                    'learning_rate_dis2': get_lr(self.optim_dis2),
                    'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))


            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] * 5 + \
                       list(metrics) + [elapsed_time] + ['best model epoch: %d' % self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.domain_loaderS)))
            if training:
                self.model_gen.train()
                self.model_dis.train()
                self.model_dis2.train()


    def train_epoch(self):
        source_domain_label = 0
        target_domain_label = 1
        smooth = 1e-7
        self.model_gen.train()
        self.model_dis.train()
        self.model_dis2.train()
        self.running_seg_loss = 0.0
        self.running_adv_loss = 0.0
        self.running_dis_diff_loss = 0.0
        self.running_dis_same_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0
        loss_adv_diff_data = 0
        loss_D_same_data = 0
        loss_D_diff_data = 0

        domain_t_loader = enumerate(self.domain_loaderT)
        domain_s_loader = enumerate(self.domain_loaderS)
        domain_s_loader_2 = enumerate(self.domain_loaderS_2)
        # domain_s_loader_3 = enumerate(self.domain_loaderS_3)

        start_time = timeit.default_timer()
        for batch_idx, sampleS in tqdm.tqdm(
                enumerate(self.domain_loaderS), total=len(self.domain_loaderS),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            metrics = []

            try:
                id_, sampleS_2 = next(domain_s_loader_2)
            except:
                domain_s_loader_2 = enumerate(self.domain_loaderS_2)
                id_, sampleS_2 = next(domain_s_loader_2)

            # try:
            #     id_, sampleS = next(domain_s_loader)
            # except:
            #     domain_s_loader = enumerate(self.domain_loaderS)
            #     id_, sampleS = next(domain_s_loader)

            try:
                id_, sampleT = next(domain_t_loader)
            except:
                domain_t_loader = enumerate(self.domain_loaderT)
                id_, sampleT = next(domain_t_loader)

            # if self.iteration  % 2 == 0:
            #     try:
            #         id_, sampleS = next(domain_s_loader_3)
            #     except:
            #         domain_s_loader_3 = enumerate(self.domain_loaderS_3)
            #         id_, sampleS = next(domain_s_loader_3)
            #     sampleS_2 = sampleS

            iteration = batch_idx + self.epoch * len(self.domain_loaderS)
            self.iteration = iteration

            assert self.model_gen.training
            assert self.model_dis.training
            assert self.model_dis2.training

            self.optim_gen.zero_grad()
            self.optim_dis.zero_grad()
            self.optim_dis2.zero_grad()

            # 1. train generator with random images
            for param in self.model_dis.parameters():
                param.requires_grad = False
            for param in self.model_dis2.parameters():
                param.requires_grad = False
            for param in self.model_gen.parameters():
                param.requires_grad = True

            imageS = sampleS['image'].cuda()
            source_map = sampleS['map'].cuda()
            source_boundary = sampleS['boundary'].cuda()
            # imageS_name = sampleS['img_name']
            # print(imageS_name)

            imageS_2 = sampleS_2['image'].cuda()
            # aug = [0.01, 0.05, 0.09]
            # imageS_2 = FDA_source_to_target(sampleS_2['image'], sampleT['image'], L = random.choice(aug))
            # imageS_2 = imageS_2.cuda()
            source_map_2 = sampleS_2['map'].cuda()
            source_boundary_2 = sampleS_2['boundary'].cuda()
            # imageS_name_2 = sampleS_2['img_name']
            # print(imageS_name_2)

            oS, boundaryS, low_fea, high_fea = self.model_gen(imageS)
            oS_2, boundaryS_2, low_fea_2, high_fea_2 = self.model_gen(imageS_2)
            loss_seg1 = bceloss(torch.sigmoid(oS), source_map)
            loss_seg2 = bceloss(torch.sigmoid(oS_2), source_map_2)
            # loss_seg2 = mseloss(torch.sigmoid(boundaryS), source_boundary)
            loss_seg = loss_seg1 + loss_seg2
            loss_seg.backward()

            self.running_seg_loss += loss_seg.item()
            loss_seg_data = loss_seg.data.item()
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')

            imageT = sampleT['image'].cuda()
            #target_map = sampleT['map'].cuda()
            #target_boundary = sampleT['boundary'].cuda()

            oT, boundaryT,low_fea_T, high_fea_T = self.model_gen(imageT)
            d_out = self.model_dis(prob_2_entropy(F.softmax(oT)))
            loss_adv = bce_loss(d_out,source_domain_label)

            d_out2 = self.model_dis2(prob_2_entropy(F.softmax(oT)))
            loss_adv += bce_loss(d_out2,source_domain_label)

            loss_adv = loss_adv/2
            loss_adv.backward()

            self.running_adv_diff_loss += loss_adv.item()
            loss_adv_diff_data = loss_adv.data.item()
            if np.isnan(loss_adv_diff_data):
                raise ValueError('loss is nan while training')

            # 2. train discriminator with source and target images
            for param in self.model_dis.parameters():
                param.requires_grad = True
            for param in self.model_dis2.parameters():
                param.requires_grad = True
            for param in self.model_gen.parameters():
                param.requires_grad = False

            oS = oS.detach()
            d_out = self.model_dis(prob_2_entropy(F.softmax(oS)))
            loss_d = bce_loss(d_out, source_domain_label)
            loss_d = loss_d / 2
            loss_d.backward()

            oS_2 = oS_2.detach()
            d_out2 = self.model_dis2(prob_2_entropy(F.softmax(oS_2)))
            loss_d2 = bce_loss(d_out2, source_domain_label)
            loss_d2 = loss_d2 / 2
            loss_d2.backward()

            oT = oT.detach()
            d_out = self.model_dis(prob_2_entropy(F.softmax(oT)))
            loss_d = bce_loss(d_out, target_domain_label)
            loss_d = loss_d / 2
            loss_d.backward()

            d_out2 = self.model_dis2(prob_2_entropy(F.softmax(oT)))
            loss_d2 = bce_loss(d_out2, target_domain_label)
            loss_d2 = loss_d2 / 2
            loss_d2.backward()

            self.running_dis_diff_loss += loss_d.item()
            loss_D_diff_data = loss_d.data.item()
            if np.isnan(loss_D_diff_data):
                raise ValueError('loss is nan while training')

            # if self.iteration  % 2 != 0:
        #0. content consistance module
            for param in self.model_dis.parameters():
                param.requires_grad = False
            for param in self.model_dis2.parameters():
                param.requires_grad = False
            for param in self.model_gen.parameters():
                param.requires_grad = False

            source_map = source_map.detach()
            source_map_2 = source_map_2.detach()

            prototype = compute_prototype(high_fea,source_map,0)
            prototype_2 = compute_prototype(high_fea_2,source_map_2,0)

            batch_size = prototype.size()[0]
            source_flatten = prototype.view(batch_size,-1)
            source_2_flatten = prototype_2.view(batch_size,-1)

            cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
            cosine_sim = torch.square(cosine(source_flatten, source_2_flatten))
            loss_consis_1 = 1-cosine_sim.mean()
            loss_consis_1 = Variable(loss_consis_1, requires_grad = True)

            prototype = compute_prototype(high_fea,source_map,1)
            prototype_2 = compute_prototype(high_fea_2,source_map_2,1)

            batch_size = prototype.size()[0]
            source_flatten = prototype.view(batch_size,-1)
            source_2_flatten = prototype_2.view(batch_size,-1)

            cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
            cosine_sim = torch.square(cosine(source_flatten, source_2_flatten))
            loss_consis_2 = 1-cosine_sim.mean()
            loss_consis_2 = Variable(loss_consis_2, requires_grad = True)

            loss_consis = (loss_consis_1 + loss_consis_2)/2
            loss_consis.backward()

            self.running_dis_same_loss += loss_consis.item()
            loss_D_same_data = loss_consis.data.item()
            if np.isnan(loss_D_same_data):
                raise ValueError('loss is nan while training')

            self.optim_gen.step()
            self.optim_dis.step()
            self.optim_dis2.step()

            # write image log
            if iteration % 30 == 0:
                grid_image = make_grid(
                    imageS[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/image', grid_image, iteration)
                grid_image = make_grid(
                    source_map[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/target_cup', grid_image, iteration)
                grid_image = make_grid(
                    source_map[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/target_disc', grid_image, iteration)
                grid_image = make_grid(
                    source_boundary[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/target_boundary', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(oS)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/prediction_cup', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(oS)[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/prediction_disc', grid_image, iteration)
                grid_image = make_grid(torch.sigmoid(boundaryS)[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('DomainS/prediction_boundary', grid_image, iteration)

            self.writer.add_scalar('train_gen/loss_seg', loss_seg_data, iteration)

            metrics.append((loss_seg_data, loss_adv_diff_data, loss_D_same_data, loss_D_diff_data))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration]  + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

        self.running_seg_loss /= len(self.domain_loaderS)
        self.running_adv_diff_loss /= len(self.domain_loaderS)
        self.running_dis_same_loss /= len(self.domain_loaderS)
        self.running_dis_diff_loss /= len(self.domain_loaderS)

        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, '
              ' Average advLoss: %f, Average dis_same_Loss: %f, '
              'Average dis_diff_Loss: %f,'
              'Execution time: %.5f' %
              (self.epoch, get_lr(self.optim_gen), self.running_seg_loss,
               self.running_adv_diff_loss,
               self.running_dis_same_loss, self.running_dis_diff_loss, stop_time - start_time))


    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch+1) % 100 == 0:
                _lr_gen = self.lr_gen * 0.2
                for param_group in self.optim_gen.param_groups:
                    param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr_gen', get_lr(self.optim_gen), self.epoch * (len(self.domain_loaderS)))
            # if (self.epoch+1) % self.interval_validate == 0:
            if (self.epoch + 1) % 5 == 0:
                self.validate()
        self.writer.close()
