
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
from torchvision.transforms import transforms

# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval
from networks.GAN import BoundaryDiscriminator, UncertaintyDiscriminator, OutputDiscriminator
import cv2
import torch.backends.cudnn as cudnn
import random
from tensorboardX import SummaryWriter

from perturbations.augmentations import augment_list, Augmentation, fda_choice
from perturbations.archive import remove_deplicates, policy_decoder


import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable, run_experiments

import copy
from theconf import Config as C, ConfigArgumentParser
from collections import OrderedDict
from functools import partial
from hyperopt import hp
import yaml


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

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return torch.nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def _get_path(dataset, model, tag):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs/train_fda/%s_%s_%s.pth.tar' % (dataset, model, tag))     # TODO

@ray.remote(num_gpus=1, max_calls=1)

def train_model(config, augment, cv_ratio_test, cv_fold, save_path = None):
    C.get()
    C.get().conf = config
    C.get()['aug'] = augment

    total_batch = C.get()["batch"]
    max_epoch = C.get()['epoch']

    source_domain_label = 0
    target_domain_label = 1

    result = OrderedDict()

    # 1. dataset
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train, split_idx = cv_fold)

    print(C.get()['aug'])
    if C.get()['aug'] is not None:
        dataroot_train = os.getcwd() + '/' + args.data_dir + args.source + '/train/ROIs/image'
        # dataroot_test = os.getcwd() + '/' + args.data_dir + args.source + '/test/ROIs/image'

        composed_transforms_train.transforms.insert(0, Augmentation(C.get()['aug'], dataroot_train))
        # composed_transforms_test.transforms.insert(0, Augmentation(C.get()['aug'], dataroot_test))

    db_source = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_train, split_idx = cv_fold)
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='test/ROIs', transform=composed_transforms_test)

    train_loader = DataLoader(db_train, batch_size=2, shuffle=False, num_workers=1,drop_last=True)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1,drop_last=True)
    source_loader = DataLoader(db_source, batch_size=2, shuffle=False, num_workers=1,drop_last=True)

    # 2. model
    model = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_eval = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()
    model_dis = UncertaintyDiscriminator().cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    # print('==> Loading %s model file: %s' %
    #       (model.__class__.__name__, model_file))
    # checkpoint = torch.load(model_file)
    #
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.train()
    # model_dis.train()

    # if args.dataset=="Domain2":
    #     npfilename = './results/prototype/pseudolabel_D2_0.75.npz'
    # elif args.dataset=="Domain1":
    #     npfilename = './results/prototype/pseudolabel_D1_0.75.npz'

    # npdata = np.load(npfilename, allow_pickle=True)
    # pseudo_label_dic = npdata['arr_0'].item()

    optim_gen = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99))
    optim_dis = torch.optim.SGD(model_dis.parameters(),lr=2.5e-5,momentum=0.99,weight_decay=0.0005)

    best_val_cup_dice = 0.0;
    best_val_disc_dice = 0.0;
    best_avg = 0.0

    iter_num = 0
    for epoch_num in tqdm.tqdm(range(C.get()['epoch']), ncols=70):

        model.train()
        model_dis.train()

        for batch_idx, (sample) in enumerate(train_loader):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
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

            optim_gen.zero_grad()
            optim_dis.zero_grad()

            for param in model.parameters():
                param.requires_grad = True
            for param in model_dis.parameters():
                param.requires_grad = False

            oS, _, _, _ = model(imageS)
            loss_seg_S = bceloss(torch.sigmoid(oS), source_map)
            loss_seg = loss_seg_S
            loss_seg.mean().backward()

            oT, _, _, _ = model(data)
            d_out = model_dis(F.softmax(oT))
            loss_adv = bce_loss(d_out,source_domain_label)
            loss_adv.backward()

            for param in model_dis.parameters():
                param.requires_grad = True
            for param in model.parameters():
                param.requires_grad = False

            oS = oS.detach()
            d_out = model_dis(F.softmax(oS))
            loss_d = bce_loss(d_out, source_domain_label)
            loss_d = loss_d / 2
            loss_d.backward()

            oT = oT.detach()
            d_out = model_dis(F.softmax(oT))
            loss_d = bce_loss(d_out, target_domain_label)
            loss_d = loss_d / 2
            loss_d.backward()

            optim_gen.step()
            optim_dis.step()
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

        if save_path:
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_mean_dice': best_avg,
                'best_cup_dice': best_val_cup_dice,
                'best_disc_dice': best_val_disc_dice,
            }, save_path)


        # if not os.path.exists('./logs/train_target'):
        #     os.mkdir('./logs/train_target')
        # if args.dataset == 'Domain1':
        #     savefile = './logs/train_target/' + 'D1_' + 'checkpoint_%d.pth.tar' % epoch_num
        # elif args.dataset == 'Domain2':
        #     savefile = './logs/train_target/' + 'D2_' + 'checkpoint_%d.pth.tar' % epoch_num
        # if model_save:
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'best_mean_dice': best_avg,
        #         'best_cup_dice': best_val_cup_dice,
        #         'best_disc_dice': best_val_disc_dice,
        #     }, savefile)

        print("cup: %.4f disc: %.4f avg: %.4f cup: %.4f disc: %.4f avg: %.4f" %
              (val_cup_dice, val_disc_dice, (val_cup_dice+val_disc_dice)/2.0, cup_hd, disc_hd, (cup_hd+disc_hd)/2.0))
        print("best cup: %.4f best disc: %.4f best avg: %.4f best cup: %.4f best disc: %.4f best avg: %.4f" %
              (best_val_cup_dice, best_val_disc_dice, best_avg, best_cup_hd, best_disc_hd, best_avg_hd))
        model.train()


    result['seg_loss'] = loss_seg
    result['val_cup_dice'] = best_val_cup_dice
    result['val_disc_dice'] = best_val_disc_dice
    result['val_avg_dice'] = best_avg
    result['best_cup_hd'] = best_cup_hd
    result['best_disc_hd'] = best_disc_hd
    result['best_avg_hd'] = best_avg_hd

    return C.get()['model']['type'], cv_fold, result

def eval_tta(config, augment, reporter):
    C.get()
    C.get().conf = config

    cv_ratio_test, cv_fold, save_path, dataroot = augment['cv_ratio_test'], augment['cv_fold'], augment['save_path'], augment['dataroot']
    # setup - provided augmentation rules
    C.get()['aug'] = policy_decoder(augment, augment['num_policy'], augment['num_op'])


    dataroot_test = dataroot + args.source + '/test/ROIs/image'
    composed_transforms_test.transforms.insert(0, Augmentation(C.get()['aug'], dataroot_test))

    #eval
    model_eval = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)

    if torch.cuda.is_available():
        model_eval = model_eval.cuda()
    print('==> Loading %s model file: %s' %
          (model_eval.__class__.__name__, save_path))
    checkpoint = torch.load(save_path)

    model_eval.load_state_dict(checkpoint['model_state_dict'])
    model_eval.eval()

    loaders = []
    for _ in range( augment['num_policy']):  # TODO
        db_test = DL.FundusSegmentation(base_dir=dataroot, dataset=args.source, split='test/ROIs', transform=composed_transforms_test)
        test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1,drop_last=True)
        loaders.append(iter(test_loader))

    metrics = Accumulator()

    with torch.no_grad():
        corrects = []
        for loader in loaders:
            for batch_idx, (sample) in enumerate(loader):
                data, target, img_name = sample['image'], sample['map'], sample['img_name']
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                prediction, boundary, _, _ = model_eval(data)
                prediction = torch.sigmoid(prediction)

                target_numpy = target.data.cpu()
                prediction = prediction.data.cpu()
                prediction[prediction>0.75] = 1;prediction[prediction <= 0.75] = 0

                cup_dice = dice_coefficient_numpy(prediction[:,0, ...], target_numpy[:, 0, ...])
                disc_dice = dice_coefficient_numpy(prediction[:,1, ...], target_numpy[:, 1, ...])
                avg_dice = (cup_dice + disc_dice)/2
                corrects.append(avg_dice)

        corrects = np.concatenate(corrects)
        corrects_max = np.max(corrects, axis=0).squeeze()

        metrics.add_dict({
            'correct': np.sum(corrects_max),
            'cnt': len(corrects)
        })

    # metrics = metrics / 'cnt'

    reporter(val_avg_dice=metrics['correct'], done=True)
    return metrics['correct']


if __name__ == '__main__':

    parser = ConfigArgumentParser(conflict_handler='resolve')
    # parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='./logs/train_source/best_model.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain3')
    parser.add_argument('--source', type=str, default='Domain1')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='Data/Fundus/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # model_file = args.model_file

    # initiating ray
    ray.init()

    # parameters / config
    cv_num = 2
    cv_ratio = 0.4
    copied_c = copy.deepcopy(C.get().conf)

    num_policy = 3
    num_op = 3

    # Train without augmentations
    paths = [_get_path(args.dataset, C.get()['model']['type'], 'ratio%.1f_fold%d' % (cv_ratio, i)) for i in range(cv_num)]
    reqs = [
        train_model.remote(copy.deepcopy(copied_c), C.get()['aug'], cv_ratio, i, save_path=paths[i]) for i in range(cv_num)]
    pretrain_results = ray.get(reqs)
    #for r_model, r_cv, r_dict in pretrain_results:

    # ops = augment_list(False)
    ops = fda_choice(False)
    space = {}
    for i in range(num_policy):
        for j in range(num_op):
            space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
            space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)

    final_policy_set = []
    #total_computation = 0
    reward_attr = 'val_avg_dice'

    for _ in range(1):  # run multiple times.
        for cv_fold in range(cv_num):
            name = "search_%s_%s_fold%d_ratio%.1f" % (C.get()['dataset'], C.get()['model']['type'], cv_fold, cv_ratio)
            # register_trainable(name, lambda augs, rpt: eval_tta(copy.deepcopy(copied_c), augs, pt))
            register_trainable(name, partial(eval_tta,copy.deepcopy(copied_c)))
            # algo = HyperOptSearch(space, max_concurrent=4*20, reward_attr=reward_attr)
            algo = HyperOptSearch(space, metric=reward_attr, mode = 'max')

        run_config = {'dataroot': os.getcwd() + '/' + args.data_dir, 'save_path': paths[cv_fold], 'cv_ratio_test': cv_ratio, 'cv_fold': cv_fold, 'num_op': num_op, 'num_policy': num_policy}
        results = tune.run(name, config = run_config, search_alg= algo, metric=reward_attr, mode = 'max', resources_per_trial={'gpu': 1})
        # print()

        final_policy = policy_decoder(results.get_best_config("val_avg_dice","max"), num_policy, num_op)
        final_policy = remove_deplicates(final_policy)
        final_policy_set.extend(final_policy)

        # with open("deeplab.yaml") as f:
        #      list_doc = yaml.safe_load(f)
        #
        # for sense in list_doc:
        #     sense["aug"] == final_policy_set
        #     print(sense)
        #
        # with open("data.yaml", "w") as f:
        #     yaml.dump(list_doc, f)

        # #Train with Augmentations

        num_experiments = 3
        default_path = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_default%d' % (cv_ratio, _)) for _ in range(num_experiments)]
        augment_path = [_get_path(C.get()['dataset'], C.get()['model']['type'], 'ratio%.1f_augment%d' % (cv_ratio, _)) for _ in range(num_experiments)]
        reqs = [train_model.remote(copy.deepcopy(copied_c), C.get()['aug'], 0.0, 0, save_path=default_path[_]) for _ in range(num_experiments)] + \
            [train_model.remote(copy.deepcopy(copied_c), final_policy_set, 0.0, 0, save_path=augment_path[_]) for _ in range(num_experiments)]

        tqdm_epoch = tqdm.tqdm(range(C.get()['epoch']))
        is_done = False

        # for epoch in tqdm_epoch:
        #     while True:
        #         epochs = OrderedDict()
        #         for exp_idx in range(num_experiments):
        #             try:
        #                 if os.path.exists(default_path[exp_idx]):
        #                     latest_ckpt = torch.load(default_path[exp_idx])
        #                     epochs['default_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
        #             except:
        #                 pass
        #             try:
        #                 if os.path.exists(augment_path[exp_idx]):
        #                     latest_ckpt = torch.load(augment_path[exp_idx])
        #                     epochs['augment_exp%d' % (exp_idx + 1)] = latest_ckpt['epoch']
        #             except:
        #                 pass
        #
        #         # tqdm_epoch.set_postfix(epochs)
        #         if len(epochs) == num_experiments*2 and min(epochs.values()) >= C.get()['epoch']:
        #             is_done = True
        #         if len(epochs) == num_experiments*2 and min(epochs.values()) >= epoch:
        #             break
        #         # time.sleep(10)
        #
        #     if is_done:
        #         break


        print('Getting results .......')
        # Final results
        final_results = ray.get(reqs)
        print(final_policy_set)

        for train_mode in ['default', 'augment']:
            avg = 0.
            for _ in range(num_experiments):
                r_model, r_cv, r_dict = final_results.pop(0)
                avg += r_dict['val_avg_dice']
            avg /= num_experiments
            print('[%s] average=%.4f (#experiments=%d)' % (train_mode, avg, num_experiments))
