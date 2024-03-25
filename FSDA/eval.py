import numpy as np
import torch
import argparse
import os
import glob

import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval

# from .misc import AverageMeter
# import SimpleITK as sitk
# import os.path as osp
import medpy.metric.binary as mmb
from Data.convert_mask import colorize_mask

from networks.unet import *



"""
The implementation is borrowed from: https://github.com/cchen-cc/SIFA
"""
def new_validate(model, test_list, n_classes=5, gpu=True, save_dir=""):
    # function for test by volume
    dice_list = []
    assd_list = []

    model.eval()
    for idx_file, fid in enumerate(test_list):
        _npz_dict = np.load(fid)
        data = _npz_dict['arr_0']
        label = _npz_dict['arr_1']

        print(f'testing volume {fid}')
        # if not gpu:
        #     model = model.cpu()

        metric_list = test_single_volume(data, label, model, n_classes, save_dir=save_dir, fid=fid)
        dice_list.append([d for d, asd in metric_list])
        assd_list.append([asd for d, asd in metric_list])

    dice_arr = 100 * np.reshape(dice_list, [-1, n_classes - 1])
    dice_mean = np.mean(dice_arr, axis=0)
    dice_std = np.std(dice_arr, axis=0)

    print('Dice:')
    print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.1f' % np.mean(dice_mean))

    assd_arr = np.reshape(assd_list, [-1, n_classes-1])

    assd_mean = np.mean(assd_arr, axis=0)
    assd_std = np.std(assd_arr, axis=0)

    print('ASSD:')
    print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.1f' % np.mean(assd_mean))

    return np.mean(dice_mean)


def test_single_volume(image, label, net, classes, patch_size=[256, 256], save_dir="", fid=""):
    # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(min(image.shape)):
        if 'leuda' in fid and 'org' not in fid:
            # print('ourdata')
            slice = image[:, :, ind]
        else:
            slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).repeat(1, 3, 1, 1).float().cuda()
        net.eval()

        with torch.no_grad():
            out, boundary, _  = net(input)
            # out = net(input)
            out = torch.argmax(torch.softmax(
                out, dim=1), dim=1).squeeze(0)
            # print(out.shape)
            # print(out.unique())
            out = out.cpu().detach().numpy()

            # img = colorize_mask(out)
            # img.save( 'results/mrct/' + str(ind) + '.png')

            pred = out

            # pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            if 'leuda' in fid and 'org' not in fid:
                prediction[...,ind] = pred
            else:
                prediction[ind] = pred

    # uncommand if want to calculate dice with largest connected components post-processing
    # prediction = post_process(prediction)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = mmb.dc(pred, gt)
        assd = mmb.assd(pred, gt)
        return dice, assd
    else:
        return 0, 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='./logs/CT/results/best_model.pth.tar')
    parser.add_argument('--dataset', type=str, default='Domain2')
    parser.add_argument('--source', type=str, default='Domain3')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='Data/Fundus/')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    parser.add_argument('--out',type=str,default='.logs/CT/results')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # model = UNet(n_channels=3, n_classes=5, bilinear=True).cuda()
    # model_eval = UNet(n_channels=3, n_classes=5, bilinear=True).cuda()

    model = netd.DeepLab(num_classes=5, backbone='resnet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_eval = netd_eval.DeepLab(num_classes=5, backbone='resnet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    #test
    model_eval.train()
    pretrained_dict = model.state_dict()
    model_dict = model_eval.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_eval.load_state_dict(pretrained_dict)

    test_list = glob.glob('./Data/MRCT/CT/eval' + "/*")
    print("==> Results")
    new_validate(model_eval, test_list, n_classes=5, gpu=True, save_dir=args.out)
