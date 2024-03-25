from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from glob import glob
import random

# from perturbations.augmentations import *
from torchvision.transforms import transforms
from theconf import Config as C

import torchvision.transforms as T
from Data.convert_mask import colorize_mask


class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 imageslist = None,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='refuge',
                 domain = 'source',
                 split='train',
                 testid=None,
                 transform=None,
                 set=None,
                 split_idx=None):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []

        self.image_list_fda = []
        self.image_list_gan = []
        self.split = split
        self.set = set

        self.split_idx = split_idx

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []


        self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
        print('Data directory: ',self._image_dir)
        self.imageslist = glob(self._image_dir + "/*.png")
        if split_idx is not None:
            splits = np.array_split(self.imageslist, 2)
            self.imageslist = list(splits[split_idx])
        # print('Augumentation: ', C.get()['aug'])

        for image_path in self.imageslist:
            gt_path = image_path.replace('image', 'mask')
            gt_path = gt_path.replace('png', 'npy') ###

            if self.set == 'cyclegan':
                image_path1 = image_path.replace('image','cyclegan')
                place = image_path1.find('.')
                # for index in range(15):
                image_path_gan = image_path1[:place] + '_fake_B' + image_path1[place:]
                self.image_list.append({'image': image_path_gan, 'label': gt_path, 'id': testid})

            # elif self.set == 'style':
            #     image_path3 = image_path.replace('image','style')
            #     image_path_style = image_path3.replace('png','jpg')
            #     self.image_list.append({'image': image_path_style, 'label': gt_path, 'id': testid})

            elif set == 'fda':
                image_path2 = image_path.replace('image','fda')
                place = image_path2.find('.')
                #index = random.randint(0,4)
                # for index in range(5):
                image_path_1 = image_path2[:place] + '_' + str(index) + '1' + image_path2[place:]
                self.image_list.append({'image': image_path_1, 'label': gt_path, 'id': testid})
                image_path_5 = image_path2[:place] + '_' + str(index) + '5' + image_path2[place:]
                self.image_list.append({'image': image_path_5, 'label': gt_path, 'id': testid})
                image_path_9 = image_path2[:place] + '_' + str(index) + '9' + image_path2[place:]
                self.image_list.append({'image': image_path_9, 'label': gt_path, 'id': testid})

            # elif self.set == 'slike':
            #     image_path1 = image_path.replace('image','slike')
            #     place = image_path1.find('.')
            #     image_path_gan = image_path1[:place] + '_fake_A' + image_path1[place:]
            #     self.image_list.append({'image': image_path_gan, 'label': gt_path, 'id': testid})
            #     image_path_gan = image_path1[:place] + '_rec_A' + image_path1[place:]
            #     self.image_list.append({'image': image_path_gan, 'label': gt_path, 'id': testid})
            #
            # elif self.set == 'tlike':
            #     image_path1 = image_path.replace('image','tlike')
            #     place = image_path1.find('.')
            #     image_path_gan = image_path1[:place] + '_fake_B' + image_path1[place:]
            #     self.image_list.append({'image': image_path_gan, 'label': gt_path, 'id': testid})
            #     image_path_gan = image_path1[:place] + '_rec_B' + image_path1[place:]
            #     self.image_list.append({'image': image_path_gan, 'label': gt_path, 'id': testid})


            else:
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
        #print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

        self.transform = transform
        # self._read_img_into_memory()
        # Display stats

        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = np.load(self.image_list[index]['label'])

        _target = np.argmax(_target, axis=2)
        _target = colorize_mask(_target)

        # _target.save('label.png')

        # pixVals = list(_target.getdata())
        # print(pixVals)

        # _target = Image.open(self.image_list[index]['label'])
        # if _target.mode == 'RGB':
            # _target = _target.convert('L')

        _img_name = self.image_list[index]['image'].split('/')[-1]

        # _img = self.image_pool[index]
        # _target = self.label_pool[index]
        # _img_name = self.img_name_pool[index]

        black = 0
        for pixel in _img.getdata():
            if pixel == (0, 0, 0): # if your image is RGB (if RGBA, (0, 0, 0, 255) or so
                black += 1
        width, height = _img.size

        if not black > 1/2 * (width * height):

            anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

            if self.transform is not None:
                anco_sample = self.transform(anco_sample)

            # transform = T.ToPILImage()
            # img_s = transform(anco_sample["image"])
            # img_s.save('1.png')

            return anco_sample
        else:
            return None

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode == 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'
