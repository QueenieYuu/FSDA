import numpy as np
import os
import os.path as osp
import random
import re

from PIL import Image
from glob import glob


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

def main():

    amp_trg_list = []
    pha_trg_list = []

    amp_trg_avg = []
    pha_trg_avg = []

    trg_root = 'Data/Fundus/Domain2/train/ROIs/image'
    src_root = 'Data/Fundus/Domain3/train/ROIs/image'

    #
    # trg_root = 'Data/MRCT/CT/train/image'
    # src_root = 'Data/MRCT/MR/train/image'

    targetlist = glob(trg_root + "/*.png")
    for index in range(5):
        randomlist = random.sample(targetlist, 10)
        for image_path in randomlist:

            im_trg = Image.open(image_path)
            im_trg = im_trg.resize( (800,800), Image.BICUBIC )
            im_trg = np.asarray(im_trg, np.float32)
            im_trg = im_trg.transpose((2, 0, 1))
            fft_trg_np = np.fft.fft2( im_trg, axes=(-2, -1) )
            amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

            amp_trg_list.append(amp_trg)
            pha_trg_list.append(pha_trg)

        amp_trg_avg.append(sum(amp_trg_list)/len(amp_trg_list))
        pha_trg_avg.append(sum(pha_trg_list)/len(pha_trg_list))

        amp_trg_list.clear()
        pha_trg_list.clear()

    sourcelist = glob(src_root + "/*.png")

    for image_path in sourcelist:
        print(image_path)

        im_src = Image.open(image_path)
        im_src = im_src.resize( (800,800), Image.BICUBIC)
        im_src = np.asarray(im_src, np.float32)
        im_src = im_src.transpose((2, 0, 1))

        fft_src_np = np.fft.fft2( im_src, axes=(-2, -1) )
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

        # 0.6185, 0.8884, 0.7963 #82.24 eval + no GAN
        # 0.2702, 0.0952, 0.2390 #83.28 seg + Gan
        # 0.6032, 0.7191, 0.8922 #81.29 seg + no GAN

        #0.3542, 0.8431, 0.6222 #D2
        #0.2067, 0.3439, 0.7790 #D1

        for index in range(5):
        # index = 0

            place1 = [m.start() for m in re.finditer('/', image_path)]
            place2 = image_path.find('.')

            save_file_1 = image_path[:place1[4]] +'/fda' + image_path[place1[5]:place2] + '_' + str(index) + '1' + image_path[place2:]
            # mutate the amplitude part of source with target
            amp_src_1 = low_freq_mutate_np(amp_src, amp_trg_avg[index], L = 0.01)
            # mutated ft of source
            fft_src_1 = amp_src_1 * np.exp( 1j * pha_src )
            # get the mutated image
            src_in_trg_1 = np.fft.ifft2( fft_src_1, axes=(-2, -1) )
            src_in_trg_1 = np.real(src_in_trg_1)
            src_in_trg_1 = src_in_trg_1.transpose((1,2,0))
            toimage(src_in_trg_1, cmin=0.0, cmax=255.0).save(save_file_1)

            save_file_5 = image_path[:place1[4]] +'/fda' + image_path[place1[5]:place2] + '_' + str(index) + '5' + image_path[place2:]
            # mutate the amplitude part of source with target
            amp_src_5 = low_freq_mutate_np( amp_src, amp_trg_avg[index], L = 0.05)
            # mutated fft of source
            fft_src_5 = amp_src_5 * np.exp( 1j * pha_src )
            # get the mutated image
            src_in_trg_5 = np.fft.ifft2( fft_src_5, axes=(-2, -1) )
            src_in_trg_5 = np.real(src_in_trg_5)
            src_in_trg_5 = src_in_trg_5.transpose((1,2,0))
            toimage(src_in_trg_5, cmin=0.0, cmax=255.0).save(save_file_5)

            save_file_9 = image_path[:place1[4]] +'/fda' + image_path[place1[5]:place2] + '_' + str(index) + '9' + image_path[place2:]
            # mutate the amplitude part of source with target
            amp_src_9 = low_freq_mutate_np( amp_src, amp_trg_avg[index], L = 0.09)
            # mutated fft of source
            fft_src_9 = amp_src_9 * np.exp( 1j * pha_src )
            # get the mutated image
            src_in_trg_9 = np.fft.ifft2( fft_src_9, axes=(-2, -1) )
            src_in_trg_9 = np.real(src_in_trg_9)
            src_in_trg_9 = src_in_trg_9.transpose((1,2,0))
            toimage(src_in_trg_9, cmin=0.0, cmax=255.0).save(save_file_9)

if __name__ == '__main__':
    main()
