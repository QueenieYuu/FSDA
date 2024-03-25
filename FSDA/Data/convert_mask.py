import numpy as np
import cv2
import os
from PIL import Image


# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156,
#            190, 153, 153, 153, 153, 153, 250,
#            170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152,
#            70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0,
#            142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
palette = [255, 255, 255, 102, 178, 255, 255, 178, 102, 255, 153, 204, 204, 255, 153]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

if __name__ == '__main__':
    # instance-level annotations -> binary annotations

    ins_folder = 'MRCT/MR/train/mask'
    out_ann_folder = 'MRCT/MR/train/mask_new'


    try:
        os.stat(os.path.dirname(out_ann_folder + '/'))
    except:
        os.mkdir(os.path.dirname(out_ann_folder + '/'))

    imglist = os.listdir(ins_folder)

    for img_name in imglist:
        img_path = os.path.join(ins_folder, img_name)
        img = np.load(img_path)

        img = np.argmax(img, axis=2)

        img = colorize_mask(img)
        img.save(os.path.join(out_ann_folder, img_name.split('.')[0] + '.png'))
        # img = Image.open(img_path) #for example image size : 28x28x3
        #print(img.size)
        # img1 = img.convert('L')  #convert a gray scale
        # print(img1.size)
        # img1.save(os.path.join(out_ann_folder, img_name.split('.')[0] + '.png'))
