
#!/usr/bin/env python

import argparse
import os
import os.path as osp

import numpy as np
import copy
import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision.transforms import transforms

# from scipy.misc import imsave
# from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval
# import cv2
import torch.backends.cudnn as cudnn
import random
from tensorboardX import SummaryWriter

bceloss = torch.nn.BCELoss(reduction='none')
choice=lambda x:x[np.random.randint(len(x))] if isinstance(x,tuple) else choice(tuple(x))

class search_config:
    blocks_keys = np.arange(0, 10).tolist()
    nr_layer=4
    states=[10]*nr_layer
    max_epochs = 1
    select_num = 10
    population_num = 5
    mutation_num = 20
    m_prob = 0.1
    crossover_num = 20


class EvolutionTrainer(object):
    def __init__(self):

        self.candidates = torch.Tensor([[-1]*search_config.nr_layer]*search_config.population_num).long().cuda()
        self.vis_dict = {}
        self.memory = []
        self.epoch = 0

        self.keep_top_k = {search_config.select_num:[],50:[]}

        self.results_scale_baseline = [0.204, 0.394, 0.483, 5.0, 4.0, 3.0]
        # self.prob_idx = [0, 2] + [(5 + 3*i) for i in range(2*5)]
        self.prob_idx = [0, 2]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands=[random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand]={}
                info=self.vis_dict[cand]

            for cand in cands:
                yield cand

    def random_can(self, num):
        # self.logger.info('random select ........')
        candidates = []
        cand_iter=self.stack_random_cand(
            lambda:tuple(np.random.randint(i) for i in search_config.states))
        while len(candidates)<num:
            cand=next(cand_iter)

            candidates.append(cand)
            # self.logger.info('random {}/{}'.format(len(candidates),num))

        # self.logger.info('random_num = {}'.format(len(candidates)))
        return candidates

    def update_top_k(self,candidates,*,k,key,reverse=False):
        assert k in self.keep_top_k
        # self.logger.info('select ......')
        t=self.keep_top_k[k]
        t+=candidates
        t.sort(key=key,reverse=reverse)
        self.keep_top_k[k]=t[:k]

    def evaluate(self, cand):

        # result = OrderedDict()

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

        # 1. dataset
        db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train)
        db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)
        db_source = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_train)

        train_loader = DataLoader(db_train, batch_size=2, shuffle=False, num_workers=1,drop_last=True)
        test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1,drop_last=True)
        source_loader = DataLoader(db_source, batch_size=2, shuffle=False, num_workers=1,drop_last=True)

        # 2. model
        model = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
        model_eval = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

        if torch.cuda.is_available():
            model = model.cuda()
        model.train()

        optim_gen = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99))
        best_val_cup_dice = 0.0;
        best_val_disc_dice = 0.0;
        best_avg = 0.0

        iter_num = 0
        for epoch_num in tqdm.tqdm(range(1), ncols=70):
            model.train()

            for batch_idx, (sample) in enumerate(train_loader):
                data, target, img_name = sample['image'], sample['map'], sample['img_name']
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                prediction, _, _, _ = model(data)

                # pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
                # pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()

                #
                try:
                    id_, sampleS = next(domain_s_loader)
                except:
                    domain_s_loader = enumerate(source_loader)
                    id_, sampleS = next(domain_s_loader)
                imageS, source_map, imageS_name = sampleS['image'], sampleS['map'], sampleS['img_name']

                if torch.cuda.is_available():
                    imageS, source_map = imageS.cuda(), source_map.cuda()
                imageS, source_map = Variable(imageS), Variable(source_map)

                oS, _, _, _ = model(imageS)

                for param in model.parameters():
                    param.requires_grad = True
                optim_gen.zero_grad()

                loss_seg_S = bceloss(torch.sigmoid(oS), source_map)
                # loss_seg_T = bceloss(torch.sigmoid(prediction), pseudo_label)

                loss_seg = loss_seg_S
                loss_seg.mean().backward()
                optim_gen.step()
                iter_num = iter_num + 1

            #eval
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

            print("cup: %.4f disc: %.4f avg: %.4f cup: %.4f disc: %.4f avg: %.4f" %
                  (val_cup_dice, val_disc_dice, (val_cup_dice+val_disc_dice)/2.0, cup_hd, disc_hd, (cup_hd+disc_hd)/2.0))
            print("best cup: %.4f best disc: %.4f best avg: %.4f best cup: %.4f best disc: %.4f best avg: %.4f" %
                  (best_val_cup_dice, best_val_disc_dice, best_avg, best_cup_hd, best_disc_hd, best_avg_hd))
            model.train()

        # result['val_cup_dice'] = best_val_cup_dice
        # result['val_disc_dice'] = best_val_disc_dice
        # result['val_avg_dice'] = best_avg
        # result['best_cup_hd'] = best_cup_hd
        # result['best_disc_hd'] = best_disc_hd
        # result['best_avg_hd'] = best_avg_hd

        result = [best_val_cup_dice, best_val_disc_dice, best_avg, best_cup_hd, best_disc_hd, best_avg_hd]

        return loss_seg, result

    def get_mutation(self,k, mutation_num, m_prob):
        assert k in self.keep_top_k
        # self.logger.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num*10

        def random_func():
            cand=list(choice(self.keep_top_k[k]))
            for i in range(len(search_config.states)):
                if np.random.random_sample()<m_prob:
                    cand[i]=np.random.randint(search_config.states[i])
            return tuple(cand)

        cand_iter=self.stack_random_cand(random_func)
        while len(res)<mutation_num and max_iters>0:
            cand=next(cand_iter)
            res.append(cand)
            max_iters-=1
        return res

    def get_crossover(self,k, crossover_num):
        assert k in self.keep_top_k
        # self.logger.info('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num
        def random_func():
            p1=choice(self.keep_top_k[k])
            p2=choice(self.keep_top_k[k])
            return tuple(choice([i,j]) for i,j in zip(p1,p2))
        cand_iter=self.stack_random_cand(random_func)
        while len(res)<crossover_num and max_iters>0:
            cand=next(cand_iter)
            res.append(cand)
            max_iters-=1

        # self.logger.info('crossover_num = {}'.format(len(res)))
        return res

    def train(self):

        # if not self.load_checkpoint():
        self.candidates = self.random_can(search_config.population_num)
            # self.save_checkpoint()

        while self.epoch<search_config.max_epochs:

            if isinstance(self.candidates, list):
                self.candidates = torch.Tensor(self.candidates).long().cuda()
            self.candidates = [tuple(cand.tolist()) for cand in self.candidates]

            losses = []
            results = []
            for cand in self.candidates:
                #
                # cfg.AUTOAUG.LIST = cand
                loss, result = self.evaluate(cand)
                losses.append(loss)
                results.append(result)

            # self.logger.info('Evaluation finish')

            for i, cand in enumerate(self.candidates):
                loss_aug = copy.deepcopy(losses[i].detach())
                loss_aug /= loss_aug.sum()
                err = loss.std()
                for j, result_s in enumerate(self.results_scale_baseline):
                    if results[i][j] < result_s:
                        #self.logger.info('Punishment for sarcrificing other scales : %s (baseline: %s) in %d th scale of %s.'%(str(copy.deepcopy(results_scales[i])), str(self.results_scale_baseline), j, str(cand)))
                        err *= (result_s/results[i][j])

                # A regularization to avoid probabilities decay to zero.
                l_prob = (9 - np.array(cand)[self.prob_idx].mean()) * 1e-2
                err += l_prob
                self.vis_dict[cand]['err'] = err
                self.vis_dict[cand]['loss'] = str(losses[i])

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
                self.vis_dict[cand]['visited'] = True

            self.update_top_k(self.candidates,k=search_config.select_num,key=lambda x:self.vis_dict[x]['err'])
            self.update_top_k(self.candidates,k=50,key=lambda x:self.vis_dict[x]['err'] )

            for i,cand in enumerate(self.keep_top_k[50]):
                ops = [search_config.blocks_keys[i] for i in cand]

            mutation = self.get_mutation(search_config.select_num, search_config.mutation_num, search_config.m_prob)
            crossover = self.get_crossover(search_config.select_num,search_config.crossover_num)
            rand = self.random_can(search_config.population_num - len(mutation) -len(crossover))

            self.candidates = mutation+crossover+rand

            print(self.candidates)
            print(self.keep_top_k[search_config.select_num])


if __name__ == '__main__':

    # parser = ConfigArgumentParser(conflict_handler='resolve')
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

    trainer=EvolutionTrainer()

    trainer.train()
