import numpy as np
import torch
import cv2
import albumentations as al

# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import os

from PIL import Image
from glob import glob

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose

random_mirror = True


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tobytes()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

def FDA(im_src, dataroot, v):

    assert 0 <= v <= 1

    amp_trg_list = []
    pha_trg_list = []

    targetlist = glob(dataroot + "/*.png")
    targetlist = random.sample(targetlist, 1)

    for image_path in targetlist:

        im_trg = Image.open(image_path)
        im_trg = im_trg.resize((800,800), Image.BICUBIC)
        im_trg = np.asarray(im_trg, np.float32)
        im_trg = im_trg.transpose((2, 0, 1))
        fft_trg_np = np.fft.fft2( im_trg, axes=(-2, -1))
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    amp_trg_list.append(amp_trg)
    pha_trg_list.append(pha_trg)

    amp_trg_avg = sum(amp_trg_list)/len(amp_trg_list)
    pha_trg_avg = sum(pha_trg_list)/len(pha_trg_list)

    amp_trg_list.clear()
    pha_trg_list.clear()

    im_src = im_src.resize( (800,800), Image.BICUBIC)
    im_src = np.asarray(im_src, np.float32)
    im_src = im_src.transpose((2, 0, 1))
    fft_src_np = np.fft.fft2( im_src, axes=(-2, -1) )
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    amp_src = low_freq_mutate_np(amp_src, amp_trg_avg, L = v)

    # mutated ft of source
    fft_src = amp_src * np.exp( 1j * pha_src )
    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)
    src_in_trg = src_in_trg.transpose((1,2,0))

    return toimage(src_in_trg, cmin=0.0, cmax=255.0)

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):

    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    # fft_src = torch.fft.rfftn(src_img.clone(),[2,2], dim=[-2,-1])
    # fft_trg = torch.fft.rfftn(trg_img.clone())

    fft_src = torch.view_as_real(torch.fft.fft2(src_img.clone()))
    fft_trg = torch.view_as_real(torch.fft.fft2(trg_img.clone()))

    #print(src_img)
    #print(fft_src)

    # get fft of both source and target
    # fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False)
    # fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()

    fft_src_ = torch.view_as_complex(fft_src_)
    # src_in_trg = torch.irfft(fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )
    src_in_trg = torch.fft.ifft2(fft_src_)
    src_in_trg = src_in_trg.real

    return src_in_trg

def fda_choice(other_aug = False):
    l = [(FDA, 0, 1), ]

    if other_aug:
        l += [
            (ShearX, -0.3, 0.3),  # 0
            (ShearY, -0.3, 0.3),  # 1
            (TranslateX, -0.45, 0.45),  # 2
            (TranslateY, -0.45, 0.45),  # 3
            (Rotate, -30, 30),  # 4
            (AutoContrast, 0, 1),  # 5
            (Invert, 0, 1),  # 6
            (Equalize, 0, 1),  # 7
            (Solarize, 0, 256),  # 8
            (Posterize, 4, 8),  # 9
            (Contrast, 0.1, 1.9),  # 10
            (Color, 0.1, 1.9),  # 11
            (Brightness, 0.1, 1.9),  # 12
            (Sharpness, 0.1, 1.9),  # 13
            (Cutout, 0, 0.2),  # 14
            # (SamplePairing(imgs), 0, 0.4),  # 15
        ]


    return l

def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l

# augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}
augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in fda_choice()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, dataroot, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), dataroot,  level * (high - low) + low)

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_augmentation():
    return al.Compose([
        al.RandomResizedCrop(512, 512, scale=(0.2, 1.)),
        al.Compose([
            # NOTE: RandomBrightnessContrast replaces ColorJitter
            al.RandomBrightnessContrast(p=1),
            al.HueSaturationValue(p=1),
        ], p=0.8),
        al.ToGray(p=0.2),
        al.GaussianBlur(5, p=0.5),
    ])


def augment(images, labels, aug):
    """Augments both image and label. Assumes input is a PyTorch tensor with
       a batch dimension and values normalized to N(0,1)."""

    # Transform label shape: B, C, W, H ==> B, W, H, C
    labels_are_3d = (len(labels.shape) == 4)
    if labels_are_3d:
        labels = labels.permute(0, 2, 3, 1)

    # Transform each image independently. This is slow, but whatever.
    aug_images, aug_labels = [], []
    for image, label in zip(images, labels):

        # Step 1: Undo normalization transformation, convert to numpy
        image = cv2.cvtColor(image.numpy().transpose(
            1, 2, 0) + IMG_MEAN, cv2.COLOR_BGR2RGB).astype(np.uint8)
        label = label.numpy()  # convert to np

        # Step 2: Perform transformations on numpy images
        data = aug(image=image, mask=label)
        image, label = data['image'], data['mask']

        # Step 3: Convert back to PyTorch tensors
        image = torch.from_numpy((cv2.cvtColor(image.astype(
            np.float32), cv2.COLOR_RGB2BGR) - IMG_MEAN).transpose(2, 0, 1))
        label = torch.from_numpy(label)
        if not labels_are_3d:
            label = label.long()

        # Add to list
        aug_images.append(image)
        aug_labels.append(label)

    # Stack
    images = torch.stack(aug_images, dim=0)
    labels = torch.stack(aug_labels, dim=0)

    # Transform label shape back: B, C, W, H ==> B, W, H, C
    if labels_are_3d:
        labels = labels.permute(0, 3, 1, 2)
    return images, labels


class Augmentation(object):
    def __init__(self, policies, dataroot):
        self.policies = policies
        self.dataroot = dataroot

    def __call__(self, sample):
        img = sample['image']
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                sample['image'] = apply_augment(img, self.dataroot, name, level)
        return sample
