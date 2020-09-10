"""
Cathay Car Damage Flow
"""
import os
import sys
import time
import logging
import numpy as np
import random
import cv2

import torch
import torchvision

import skimage.transform

import cathay_utils

from obj_detection import mask_rcnn
from obj_detection import utils as obj_utils

from distutils.version import LooseVersion

logging.basicConfig(level=logging.DEBUG)


def gray_3_ch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _timg = np.zeros((gray.shape[0], gray.shape[1], 3))
    _timg[:, :, 0] = gray
    _timg[:, :, 1] = gray
    _timg[:, :, 2] = gray
    return _timg


def resize_image(img, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    # Keep track of image dtype and return results in the same dtype
    image_dtype = img.dtype
    logging.debug("<resize_image> dtype:{}".format(image_dtype))
    # Default window (x1, y1, x2, y2) and default scale == 1.
    h, w = img.shape[:2]
    if w <= 0:
        print('Error:resize_image:w<=0')
        raise ValueError('Error:resize_image:w<=0')

    if h <= 0:
        print('Error:resize_image:h<=0')
        raise ValueError('Error:resize_image:h<=0')

    window = (0, 0, w, h)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]

    if mode == "none":
        return img, window, scale, padding

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        img = resize(img, (round(h * scale), round(w * scale)), preserve_range=True)
        # image = skimage.transform.resize(image, (round(h * scale), round(w * scale)),
        #        order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = img.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        img = np.pad(img, padding, mode='constant', constant_values=0)
        window = (left_pad, top_pad, w + left_pad, h + top_pad)
    else:
        raise Exception("Mode {} not supported".format(mode))

    return img.astype(image_dtype), window, scale, padding


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
            preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


class ImgSet(object):
    def __init__(self, imgs,
                 resize_img=[False, 224, 224],
                 padding=[True, 32],
                 transforms=None):
        self.imgs = imgs
        self.is_resize = resize_img[0]
        self.is_padding = padding[0]
        self.min_dim = resize_img[1]
        self.max_dim = resize_img[2]
        self.padding = padding[1]
        self.transforms = transforms

    def __getitem__(self, idx):
        img = self.imgs[idx].copy()
        img = img.astype(np.uint8)

        if self.is_padding:
            padding = [(self.padding, self.padding), (self.padding, self.padding), (0, 0)]
            img_p = np.pad(img, padding, mode='constant', constant_values=0)

        if self.is_resize:
            img_p_r, window, scale, resize_padding = resize_image(img_p,
                                                                  min_dim=self.min_dim,
                                                                  max_dim=self.max_dim)

        if self.transforms is not None:
            img_torch = self.transforms(img_p_r.copy())

        return img_torch

    def __len__(self):
        return len(self.imgs)

class ObjSet(object):
    def __init__(self,
                 imgs,
                 resize_img=[False, 512, 512],
                 padding=[True, 32],
                 transforms=None):
        self.imgs = imgs
        self.is_resize = resize_img[0]
        self.is_padding = padding[0]
        self.min_dim = resize_img[1]
        self.max_dim = resize_img[2]
        self.padding = padding[1]
        self.transforms = transforms

    def __getitem__(self, idx):
        target = {}
        pimg = self.imgs[idx]
        img = cv2.imread(pimg)
        if img.shape[2] != 3:
            img_org = gray_3_ch(img)
        else:
            img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_org = img_org.astype(np.uint8)

        if self.is_padding:
            padding = [(self.padding, self.padding), (self.padding, self.padding), (0, 0)]
            img_p = np.pad(img_org, padding, mode='constant', constant_values=0)

        if self.is_resize:
            img_p_r, window, scale, resize_padding = resize_image(img_p,
                                                                  min_dim=self.min_dim,
                                                                  max_dim=self.max_dim)

        if self.transforms is not None:
            img_torch = self.transforms(img_p_r.copy())

        target["idx"] = torch.as_tensor(idx)
        target["window"] = torch.as_tensor(window, dtype=torch.float32)
        target["scale"] = torch.as_tensor(scale, dtype=torch.float32)
        target["padding"] = torch.as_tensor(resize_padding, dtype=torch.uint8)

        return img_torch, target

    def __len__(self):
        return len(self.imgs)


def args_check(args):
    for d in [args.case_path, args.log_path]:
        if not os.path.isdir(d):
            try:
                os.makedirs(d)
            except:
                logging.error("cannot create folder at path:{}".format(d))


@torch.no_grad()
def car_segmentation(cases):
    MODEL_PATH = "./weights/mrcnn_cd_20200908_101_aug_14.pth"
    BACKBONE = 'resnet101'
    CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR",
             "CDBL", "CFFR", "CFFL", "CFBR", "CFBL", "CC", "CP", "CL"]
    DIM = 1024
    PAD = 32
    BATCHSIZE = 16

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    imgset = ObjSet(cases,
                    resize_img=[True, DIM, DIM],
                    padding=[True, PAD],
                    transforms=transforms)

    dbloader = torch.utils.data.DataLoader(imgset, batch_size=BATCHSIZE, shuffle=False, collate_fn=obj_utils.collate_fn)


    mrcnn = mask_rcnn.MaskRCNN(backbone=BACKBONE,
                               anchor_ratios=(0.33, 0.5, 1, 2, 3),
                               num_classes=1 + len(CLASS))

    mrcnn.load_state_dict(torch.load(MODEL_PATH))
    mrcnn.to(device)

    mrcnn.eval()
    metric_logger = obj_utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    class_names = ['BG'] + CLASS
    for images, targets in metric_logger.log_every(dbloader, dbloader.batch_size, header):
        images = list(img.to(device) for img in images)
        torch.cuda.synchronize()
        outputs = mrcnn(images)
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        #print(outputs)
        print(targets)


def plate_detection():
    pass

def color_detection():
    pass

def logo_detection():
    pass



if __name__ == '__main__':
    args = cathay_utils.parse_commands()

    args_check(args)
    case_files = []
    for rs, ds, fs in os.walk(args.case_path):
        if os.name == 'nt':
            folder = cathay_utils.nt_path(rs)
        else:
            folder = rs
        for f in fs:
            fpath = cathay_utils.path_join(folder, f)
            case_files.append(fpath)

    logging.debug("files in case:{}".format(case_files))

    car_segmentation(case_files)