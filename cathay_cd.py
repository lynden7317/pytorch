"""
Cathay Car Damage Flow
"""
import os
import psutil
import sys
import datetime
import requests
import time
import copy
import math
import logging
import numpy as np
import cv2
import json

import torch
import torchvision

import skimage.transform

import cathay_utils

from obj_detection import mask_rcnn
from obj_detection import utils as obj_utils
from classification import models

from distutils.version import LooseVersion

logging.basicConfig(level=logging.DEBUG)  # logging.DEBUG

IMGDICT = {}
NP_WHERE_MASK = 0.5

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
        img_org = self.imgs[idx]
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

def smoothing(mask, path):
    # parameters
    blur_kernel = (7, 7)
    epsilon = 0.01
    # masks smoothing
    logging.debug("<mask smoothing> shape: {}".format(mask.shape))
    print(len(np.where(mask > NP_WHERE_MASK)[0]), len(np.where(mask > 0)[0]))
    org_mask = np.zeros((mask.shape[0],mask.shape[1]))
    smooth_mask = np.zeros((mask.shape[0],mask.shape[1]))
    cols, rows = np.where(mask > NP_WHERE_MASK)
    print("before:{}".format(len(cols)))
    org_mask[cols, rows] = 255
    org_mask = cv2.blur(org_mask, blur_kernel)
    org_mask = org_mask.astype(np.uint8)
    try:
        cnts, hier = cv2.findContours(org_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(cnts))
        # filter the max cnt
        max_cnts = None
        max_peri = 0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            if peri > max_peri:
                max_peri = peri
                max_cnts = c
        approx = cv2.approxPolyDP(max_cnts, epsilon*max_peri, True)
        cv2.drawContours(smooth_mask, [approx], -1, (255), -1)
        #smooth_mask = smooth_mask/255
        cols, rows = np.where(smooth_mask > 1)
        x1, y1 = rows.min()-1, cols.min()-1
        x2, y2 = rows.max()+1, cols.max()+1
        w, h = x2-x1, y2-y1
        print("after:{}".format(len(cols)))
        print(len(np.where(smooth_mask>0)[0]))
        cv2.imwrite(path, smooth_mask)
        #sys.exit(1)
        return True, (x1,y1,w,h,cols,rows)
    except:
        logging.warning("<Warning smoothing> cannot do the smoothing!")
        return False, (-1,-1,-1,-1,[],[])

    #for c in cnts:
    #    peri = cv2.arcLength(c, True)
    #    approx = cv2.approxPolyDP(c, epsilon*peri, True)
    #    smooth_mask = cv2.drawContours(smooth_mask, [approx], -1, (255, 255, 255), -1)

def seg_merge(results):
    _ll = list(set(results["labels"]))
    diffs = len(_ll)
    merge_boxes = np.zeros((diffs, 4))
    merge_scores = np.zeros(diffs)
    merge_masks = np.zeros((diffs, results["masks"].shape[1],
                            results["masks"].shape[2], results["masks"].shape[3]))
    print(results["labels"], diffs, results["boxes"].shape, results["scores"].shape, results["masks"].shape)
    try:
        for i, v in enumerate(results["labels"]):
            idx = _ll.index(v)
            merge_masks[idx] = merge_masks[idx] + results["masks"][i]
        for i in range(merge_masks.shape[0]):
            cols, rows = np.where(merge_masks[i,:,:,0] > NP_WHERE_MASK)
            x1, y1 = rows.min(), cols.min()
            x2, y2 = rows.max(), cols.max()
            merge_boxes[i] = np.array([x1,y1,x2,y2])
            merge_scores[i] = -1
        return {"boxes": merge_boxes, "labels": _ll, "scores": merge_scores, "masks": merge_masks}
    except:
        logging.warning("<Warning seg_merge> cannot merge the segmentations!")
        return results

def seg_criterion(tar, thresholds):
    _boxes = tar['boxes'].cpu().clone().numpy()
    _labels = tar['labels'].cpu().clone().numpy()
    _scores = tar['scores'].cpu().clone().numpy()
    _masks = tar['masks'].cpu().clone().numpy()
    _masks = _masks.transpose((0, 2, 3, 1))

    # ==== update results with score_threshold ==== #
    scores_pass = np.where(_scores > thresholds['score'])
    boxes = _boxes[scores_pass]
    labels = _labels[scores_pass]
    scores = _scores[scores_pass]
    masks = _masks[scores_pass]

    # ==== update results with nms_threshold ==== #
    _nms_result = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), thresholds['nms'])
    nms_result = _nms_result.numpy()
    #logging.debug("nms_result: {}".format(nms_result))
    boxes = boxes[nms_result,:]
    labels = labels[nms_result]
    scores = scores[nms_result]
    masks = masks[nms_result,:,:,:]

    return {"boxes":boxes, "labels":labels, "scores":scores, "masks":masks}

def major_car(cls_dict):
    # determine the major car segmentations
    pass

def mask_decode(image, result, classes, outputfolder,
                tag='c',
                is_smooth=False):
    cls_dict = {}
    cls_list = [[] for c in classes]
    path = result["path"]
    img_name = os.path.basename(path).split(".")[0]
    img = image.cpu().clone()
    img = torchvision.transforms.ToPILImage()(img)
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _name = img_name+".jpg"
    path_org = os.path.join(outputfolder, _name)
    if path_org not in IMGDICT.keys():
        IMGDICT[path_org] = img_bgr

    cv2.imwrite(path_org, img_bgr)

    labels = result["labels"]
    boxes = result["boxes"]
    masks = result["masks"][:,:,:,0]
    logging.debug("<car seg.> mask:{}, boxes:{}".format(masks.shape, boxes.shape))

    if logging.getLogger().level == logging.DEBUG:
        all_mask_img = np.zeros(img_bgr.shape)

    for _m in range(masks.shape[0]):
        lab = labels[_m]

        if tag == 'd':
            # damage area criteria
            pos = np.where(masks[_m,:,:] > NP_WHERE_MASK)
            logging.debug("damage pixels:{}, class:{}".format(pos[0].shape[0], classes[lab]))
            if pos[0].shape[0] < 4000 and classes[lab] in ['DS', 'DD', 'DC', 'DW']:
                # the damage region is too small
                continue

        x1,y1,x2,y2 = math.floor(boxes[_m, 0]), math.floor(boxes[_m, 1]), \
                      math.ceil(boxes[_m, 2]), math.ceil(boxes[_m, 3])
        w, h = x2-x1, y2-y1
        logging.debug("<car box> (x1,y1):{}, (w,h):{}".format((x1,y1), (w,h)))

        _name = img_name + "_" + tag + "_" + str(lab) + "_" + str(len(cls_list[lab])) + ".jpg"
        path = os.path.join(outputfolder, _name)
        if logging.getLogger().level == logging.DEBUG:
            mask_img = np.zeros(img_bgr.shape)
            for c in range(3):
                all_mask_img[:,:,c] = np.where(masks[_m,:,:] > NP_WHERE_MASK, img_bgr[:,:,c], all_mask_img[:,:,c])
                mask_img[:,:,c] = np.where(masks[_m,:,:] > NP_WHERE_MASK, img_bgr[:,:,c], 0)

            mask_img = mask_img[y1:y1+h,x1:x1+w,:]
            cv2.imwrite(path, mask_img)

        _name = img_name + "_" + tag + "_" + str(lab) + "_" + str(len(cls_list[lab])) + "_s.jpg"
        path = os.path.join(outputfolder, _name)

        if is_smooth:
            flag, updated = smoothing(masks[_m,:,:], path)
            if flag:
                x1, y1, w, h, cols, rows = updated
        else:
            cols, rows = np.where(masks[_m,:,:] > NP_WHERE_MASK)

        cls_list[lab].append((path_org, path, (x1,y1), (w,h), (cols,rows)))

    #logging.debug("predict list:{}".format(cls_list))

    if logging.getLogger().level == logging.DEBUG:
        _name = img_name+"_" + tag + ".jpg"
        cv2.imwrite(os.path.join(outputfolder, _name), all_mask_img)

    for i, c in enumerate(classes):
        cls_dict[c] = cls_list[i]

    #logging.debug("predict dict:{}".format(cls_dict))
    return cls_dict

def car_model(device):
    MODEL_PATH = "./weights/mrcnn_cd_20200908_101_aug_14.pth"
    BACKBONE = 'resnet101'
    CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR",
             "CDBL", "CFFR", "CFFL", "CFBR", "CFBL", "CC", "CP", "CL"]

    mrcnn = mask_rcnn.MaskRCNN(backbone=BACKBONE,
                               anchor_ratios=(0.33, 0.5, 1, 2, 3),
                               num_classes=1 + len(CLASS))

    mrcnn.load_state_dict(torch.load(MODEL_PATH))
    mrcnn.to(device)
    mrcnn.eval()

    return mrcnn, CLASS

def damage_model(device):
    MODEL_PATH = "./weights/mrcnn_cd_20200821_aug_10.pth"
    BACKBONE = 'resnet50'
    CLASS = ['DS', 'DD', 'DC', 'DW', 'DH']

    mrcnn = mask_rcnn.MaskRCNN(backbone=BACKBONE,
                               anchor_ratios=(0.33, 0.5, 1, 2, 3),
                               num_classes=1 + len(CLASS))

    mrcnn.load_state_dict(torch.load(MODEL_PATH))
    mrcnn.to(device)
    mrcnn.eval()

    return mrcnn, CLASS

def merge_car_damage_segs(cars, damages, outputfolder):
    cls_dict = {}
    for _c in cars.keys():
        cls_dict[_c] = []
        if _c in ["CC", "CP", "CL"]:
            cls_dict[_c] = cars[_c]
        else:
            for _cc in cars[_c]:
                org_shape = IMGDICT[_cc[0]].shape
                c_cols, c_rows = _cc[4]
                c_pos = np.zeros((org_shape[0], org_shape[1]))
                c_pos[c_cols, c_rows] = 1
                logging.debug("<merge_car_damage_segs> org. shape:{}".format(org_shape))
                loc_damages = {}
                for _d in damages.keys():
                    for _dd in damages[_d]:
                        logging.debug("  car:{}, damage:{}".format(_cc[1], _dd[1]))
                        d_cols, d_rows = _dd[4]
                        d_pos = np.zeros((org_shape[0], org_shape[1]))
                        d_pos[d_cols, d_rows] = 1
                        iou_pos = c_pos*d_pos
                        iou = np.sum(np.sum(iou_pos, axis=0), axis=0)
                        logging.debug("  intersect iou:{}".format(iou))
                        if iou > 0:
                            cd_cols, cd_rows = np.where(iou_pos > NP_WHERE_MASK)
                            c_pos = c_pos - iou_pos
                            loc_damages[_d] = (cd_cols, cd_rows)

                c_cols, c_rows = np.where(c_pos > NP_WHERE_MASK)
                cls_dict[_c].append((_cc[0], _cc[1], _cc[2], _cc[3], (c_cols, c_rows), loc_damages))

                if logging.getLogger().level == logging.DEBUG:
                    img_bgr = IMGDICT[_cc[0]]
                    updated_car_mask = np.zeros(org_shape)
                    updated_name = os.path.basename(_cc[1]).split(".")[0]
                    for c in range(3):
                        updated_car_mask[:, :, c] = np.where(c_pos > NP_WHERE_MASK, img_bgr[:, :, c], 0)
                    _name = updated_name + "_u.jpg"
                    cv2.imwrite(os.path.join(outputfolder, _name), updated_car_mask)

    return cls_dict

@torch.no_grad()
def car_segmentation(cases, folder):
    DIM = 1024
    PAD = 32
    BATCHSIZE = 16
    CAR_SCORETHRESHOLD = 0.7
    CAR_NMSTHRESHOLD = 0.3
    DAMAGE_SCORETHRESHOLD = 0.7
    DAMAGE_NMSTHRESHOLD = 0.3

    TMPFOLDER = os.path.join(cathay_utils.nt_path(folder), 'tmp')
    if not os.path.isdir(TMPFOLDER):
        try:
            os.makedirs(TMPFOLDER)
        except:
            logging.error("cannot create folder at path:{}".format(TMPFOLDER))
            TMPFOLDER = cathay_utils.nt_path(folder)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    imgset = ObjSet(cases,
                    resize_img=[True, DIM, DIM],
                    padding=[True, PAD],
                    transforms=transforms)

    dbloader = torch.utils.data.DataLoader(imgset, batch_size=BATCHSIZE, shuffle=False, collate_fn=obj_utils.collate_fn)

    c_mrcnn, car_classes = car_model(device)
    d_mrcnn, damage_classes = damage_model(device)

    metric_logger = obj_utils.MetricLogger(delimiter="  ")
    header = '<car & damage seg.> Eval:'
    car_class_names = ['BG'] + car_classes
    damage_class_names = ['BG'] + damage_classes
    car_return_dict = dict((k, []) for k in car_class_names)
    for images, targets in metric_logger.log_every(dbloader, dbloader.batch_size, header):
        images = list(img.to(device) for img in images)
        torch.cuda.synchronize()
        c_outputs = c_mrcnn(images)   # car segmentation
        d_outputs = d_mrcnn(images)   # damage segmentation
        c_outputs = [{k: v.to(device) for k, v in t.items()} for t in c_outputs]
        d_outputs = [{k: v.to(device) for k, v in t.items()} for t in d_outputs]
        #print(len(c_outputs), len(d_outputs))
        car_mask_dict, damage_mask_dict, car_damage_dict = None, None, None
        for _i, c in enumerate(c_outputs):
            logging.debug("car seg. target keys:{}".format(targets[_i].keys()))
            d = d_outputs[_i]
            img_idx = targets[_i]['idx'].cpu().clone().numpy()
            img_path = imgset.imgs[img_idx]

            if len(d['labels']) > 0:
                d_results = seg_criterion(d, {'score':DAMAGE_SCORETHRESHOLD, 'nms':DAMAGE_NMSTHRESHOLD})
                d_results = seg_merge(d_results)
                d_results.update({"path": img_path})
                damage_mask_dict = mask_decode(images[_i], d_results, damage_class_names, TMPFOLDER, tag='d')

            if len(c['labels']) > 0:
                c_results = seg_criterion(c, {'score':CAR_SCORETHRESHOLD, 'nms':CAR_NMSTHRESHOLD})
                c_results.update({"path": img_path})
                car_mask_dict = mask_decode(images[_i], c_results, car_class_names, TMPFOLDER,
                                            tag='c', is_smooth=True)
                major_car(car_mask_dict)

            if car_mask_dict is not None and damage_mask_dict is not None:
                logging.debug("merge car & damage segmentations")
                car_damage_dict = merge_car_damage_segs(car_mask_dict, damage_mask_dict, TMPFOLDER)

            if car_damage_dict is not None:
                for _k in car_damage_dict.keys():
                    car_return_dict[_k] += car_damage_dict[_k]

    #logging.debug("case seg. results:{}".format(class_dict))
    #sys.exit(1)
    return car_return_dict

def case_division(seg_dict):
    case_dict = {"plates":[], "logos":[], "colors":[], "damages":[]}

    for c in seg_dict.keys():
        if c in ["CC", "CP", "CL"]:
            continue
        else:
            case_dict["colors"] += copy.deepcopy(seg_dict[c])

    case_dict["logos"] = copy.deepcopy(seg_dict["CL"])

    return case_dict


def plate_detection(case_list, folder):
    DIM = 1024
    PAD = 32
    URL = "http://211.21.191.139:8080/lpr_one"

    TMPFOLDER = cathay_utils.path_join(cathay_utils.nt_path(folder), 'tmp')
    plate_return_dict = {"plates":[]}
    if not os.path.isdir(TMPFOLDER):
        try:
            os.makedirs(TMPFOLDER)
        except:
            logging.error("cannot create folder at path:{}".format(TMPFOLDER))
            return

    resize_img = [True, DIM, DIM]
    padding = [True, PAD]

    for cs in case_list:
        # location as (x, y, w, h)
        plate = {"image_path":"", "plate":"", "location":[], "confidence":0}
        img_org = cv2.imread(cs)
        if img_org.shape[2] != 3:
            img_org = gray_3_ch(img_org)

        img_org = img_org.astype(np.uint8)

        if padding[0]:
            pad_format = [(padding[1], padding[1]), (padding[1], padding[1]), (0, 0)]
            img_p = np.pad(img_org, pad_format, mode='constant', constant_values=0)

        if resize_img[0]:
            img_p_r, window, scale, resize_padding = resize_image(img_p,
                                                                  min_dim=resize_img[1],
                                                                  max_dim=resize_img[2])

        # save resizing image to FOLDER_TMP
        _path = cathay_utils.path_join(TMPFOLDER, os.path.basename(cs))
        cv2.imwrite(_path, img_p_r)
        time.sleep(1)

        files = {'file': open(_path, 'rb')}
        try:
            response = requests.post(url=URL, files=files, timeout=2)
        except requests.exceptions.ReadTimeout:
            print('Request timed out')

        print(response, response.text)
        json_format = json.loads(response.text)
        print(json_format)
        if json_format is None:
            print("No plate is detected")
            continue
        else:
            for t in json_format['tag']:
                plate["image_path"] = _path
                plate["plate"] = t['plateNumber']
                x, y, w, h = int(t['objectPicX']), int(t['objectPicY']), int(t['objectWidth']), int(t['objectHeight'])
                plate["confidence"] = t['plateConfidence']
                plate["location"] = [x, y, w, h]
                plate_return_dict["plates"].append(plate)

        #sys.exit(1)

    print("plate_return_dict: {}".format(plate_return_dict))
    return plate_return_dict


def color_detection(case_list):
    # (path_org, path, (x1,y1), (w,h), (cols, rows))
    # received color seq: BGR
    #logging.debug("colors: {}".format(case_list))
    grid_size = 10
    colors = {}
    for c in case_list:
        #logging.debug("path:{}, (w,h):{}, mask_img:{}, pos_img:{}".format(c[1], c[3], c[4].shape, c[5].shape))
        path_org = c[0]
        img_org = IMGDICT[path_org]
        path = c[1]
        x1, y1 = c[2]
        w, h = c[3]
        cols_org, rows_org = c[4]
        x = np.linspace(0, w, grid_size+1, dtype=np.int32)
        y = np.linspace(0, h, grid_size+1, dtype=np.int32)
        cols = cols_org - y1
        rows = rows_org - x1
        logging.debug("{},{},{},{},{}".format(path,(x1,y1),(w,h),cols.shape, rows.shape))

        mask_img = np.zeros((h,w,3))
        pos_mask = np.zeros((h,w))
        mask_img[cols,rows,:] = img_org[cols_org,rows_org,:]
        pos_mask[cols,rows] = 1
        for gx in range(grid_size):
            _x1, _x2 = x[gx], x[gx + 1]
            for gy in range(grid_size):
                _y1, _y2 = y[gy], y[gy+1]
                avg = np.sum(pos_mask[_y1:_y2, _x1:_x2], dtype=np.int32)
                color_bgr = np.sum(np.sum(mask_img[_y1:_y2, _x1:_x2, :], axis=0), axis=0)
                if avg > 0:
                    avg = float(1.0/float(avg))
                    color_bgr_avg = np.array(color_bgr*avg, dtype=np.int32)
                    #print(avg, color_bgr_avg)
                    # input to rgb
                    cname = cathay_utils.closest_colour((color_bgr_avg[2],color_bgr_avg[1],color_bgr_avg[0]))
                    if cname in colors.keys():
                        colors[cname] += 1
                    else:
                        colors[cname] = 1
                    #print(avg, color_bgr_avg, cname)

    max_color = sorted(colors, key=lambda k: colors[k])
    if len(max_color) > 2:
        colorName = max_color[-1] + "_" + max_color[-2]
    elif len(max_color) > 1:
        colorName = max_color[-1]
    else:
        colorName = "None"

    logging.info("<color detection> color detection: {}".format(colorName))
    return colorName


@torch.no_grad()
def logo_detection(case_list):
    logging.debug("logos: {}".format(case_list))
    MODEL_PATH = "./weights/resnet50_10.pth"
    MODEL = 'resnet50'
    CLASSES = ['Alfa Romeo', 'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Dacia', 'Daewoo', 'Dodge', 'Ferrari', 'Fiat', \
               'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Lada', 'Lancia', 'Land Rover', 'Lexus', \
               'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', \
               'Saab', 'Seat', 'Skoda', 'Subaru', 'Suzuki', 'Tata', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo']
    PAD = 8
    DIM = 128
    BATCHSIZE = 16


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    cases = []
    for i in case_list:
        img_bgr = IMGDICT[i[0]]
        x1, y1 = i[2]
        w, h = i[3]
        cols_org, rows_org = i[4]
        logo_img_bgr = np.zeros((h,w,3))
        logo_img_rgb = np.zeros((h, w, 3))
        cols = cols_org - y1
        rows = rows_org - x1
        logo_img_bgr[cols,rows,:] = img_bgr[cols_org,rows_org,:]
        logo_img_rgb[:,:,0] = logo_img_bgr[:,:,2]
        logo_img_rgb[:,:,1] = logo_img_bgr[:,:,1]
        logo_img_rgb[:,:,2] = logo_img_bgr[:,:,0]
        logging.debug("logo image shape:{}".format(logo_img_rgb.shape))
        cases.append(logo_img_rgb)

    logging.debug("<logo detection> #cases:{}".format(len(cases)))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    imgset = ImgSet(cases,
                    resize_img=[True, DIM, DIM],
                    padding=[True, PAD],
                    transforms=transforms)
    dbloader = torch.utils.data.DataLoader(imgset, batch_size=BATCHSIZE, shuffle=False)

    resnet, features = models.NNsModel(device, model_name=MODEL, classes=len(CLASSES))
    #logging.debug("resnet: {}".format(resnet))

    resnet.load_state_dict(torch.load(MODEL_PATH))
    resnet.to(device)
    logging.info("<logo detection> load pretrained weight: {}".format(MODEL_PATH))

    resnet.eval()
    metric_logger = obj_utils.MetricLogger(delimiter="  ")
    header = '<logo> Eval:'
    results = []
    for images in metric_logger.log_every(dbloader, dbloader.batch_size, header):
        images = images.to(device)
        torch.cuda.synchronize()
        outputs = resnet(images)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().clone().numpy().tolist()
        results.append(max(set(preds), key=lambda x:preds.count(x)))
        #print(preds, type(preds))

    if len(results) > 0:
        max_pred_logo = CLASSES[max(set(results), key=lambda x:results.count(x))]
    else:
        max_pred_logo = "None"
    logging.info("<logo detection> predicted logo: {}".format(max_pred_logo))
    return max_pred_logo

def summary(seg_dict, logo, color, outputfolder, is_plot=False):
    def mask2poly(pos):
        cnts, hier = cv2.findContours(pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 1:
            points = []
            for c in cnts:
                _cnt = []
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                for p in range(approx.shape[0]):
                    _cnt.append([int(approx[p][0][0]), int(approx[p][0][1])])
                points.append(_cnt)
        elif len(cnts) == 1:
            points = []
            peri = cv2.arcLength(cnts[0], True)
            approx = cv2.approxPolyDP(cnts[0], 0.01 * peri, True)
            for p in range(approx.shape[0]):
                points.append([int(approx[p][0][0]), int(approx[p][0][1])])
        else:
            points = []

        return points

    name = "summary.json"
    path = os.path.join(outputfolder, name)
    data = {"images":len(IMGDICT), "folder": outputfolder, "logo":logo, "color":color, "cars":[]}
    for seg in seg_dict.keys():
        if seg in ['BG']:
            continue
        #print(seg_dict[seg], len(seg_dict[seg]))
        for _c in seg_dict[seg]:
            part = {"label": seg, "image_path":_c[0], "points": [], "damages": []}
            if len(_c) < 6:
                org_shape = IMGDICT[_c[0]].shape
                c_cols, c_rows = _c[4]
                c_pos = np.zeros((org_shape[0], org_shape[1]))
                c_pos[c_cols, c_rows] = 255
                c_pos = c_pos.astype(np.uint8)
                points = mask2poly(c_pos)
                part["points"] = points

            else:              # including damages
                org_shape = IMGDICT[_c[0]].shape
                c_cols, c_rows = _c[4]
                c_pos = np.zeros((org_shape[0], org_shape[1]))
                c_pos[c_cols, c_rows] = 255
                #print(_c[5])
                for d in _c[5].keys():
                    d_pos = np.zeros((org_shape[0], org_shape[1]))
                    d_cols, d_rows = _c[5][d]
                    c_pos[d_cols, d_rows] = 255
                    d_pos[d_cols, d_rows] = 255
                    d_pos = d_pos.astype(np.uint8)
                    dpoints = mask2poly(d_pos)
                    part["damages"].append({"label":d, "points":dpoints})
                c_pos = c_pos.astype(np.uint8)
                points = mask2poly(c_pos)
                part["points"] = points

            data["cars"].append(part)

    json_data = json.dumps(data, indent=4, separators=(',', ': '))
    with open(path, 'w') as fid:
        fid.write(json_data)

    if is_plot:
        alpha = 0.5
        img_dict = {}
        for c in data["cars"]:
            img_path = c["image_path"]
            img = cv2.imread(img_path)
            if img_path not in img_dict.keys():
                img_dict[img_path] = [img, []]
        for c in data["cars"]:
            img_path = c["image_path"]
            mask = np.zeros((img_dict[img_path][0].shape[0], img_dict[img_path][0].shape[1]))
            points = np.array(c["points"])
            points = np.reshape(points, (points.shape[0], 1, points.shape[1]))

            # cv2.drawContours(mask, [cnt.astype(int)], -1, (255), -1)  # fill the contour
            cv2.drawContours(mask, [points.astype(int)], -1, (255), 5)

            damages = c["damages"]
            for d in damages:
                #print(len(np.asarray(d["points"]).shape))
                if len(np.asarray(d["points"]).shape) == 1:
                    # contain multiple contours
                    for cnt in d["points"]:
                        dpoints = np.array(cnt)
                        dpoints = np.reshape(dpoints, (dpoints.shape[0], 1, dpoints.shape[1]))
                        cv2.drawContours(mask, [dpoints.astype(int)], -1, (255), 5)
                elif len(np.asarray(d["points"]).shape) == 2:
                    dpoints = np.array(d["points"])
                    dpoints = np.reshape(dpoints, (dpoints.shape[0], 1, dpoints.shape[1]))
                    cv2.drawContours(mask, [dpoints.astype(int)], -1, (255), 5)
                else:
                    pass

            img_dict[img_path][1].append(mask)

        for k in img_dict.keys():
            basename = os.path.basename(k)
            img = img_dict[k][0]
            N = len(img_dict[k][1])
            colors = cathay_utils.random_colors(N)
            for i in range(N):
                color = colors[i]
                mask = img_dict[k][1][i]
                for c in range(3):
                    img[:, :, c] = np.where(mask >= 0.5,
                                            img[:, :, c] *(1 - alpha) + alpha * color[c] * 255,
                                            img[:, :, c])
            newname = "predict_"+basename
            outpath = os.path.join(data["folder"], newname)
            cv2.imwrite(outpath, img)


if __name__ == '__main__':
    args = cathay_utils.parse_commands()

    args_check(args)
    case_files = []
    for rs, ds, fs in os.walk(args.case_path):
        if os.name == 'nt':
            folder = cathay_utils.nt_path(rs)
        else:
            folder = rs
        if "tmp" in folder:
            continue
        for f in fs:
            if "json" in f:
                continue
            if "predict" in f:
                continue
            fpath = cathay_utils.path_join(folder, f)
            case_files.append(fpath)

    logging.debug("files in case:{}".format(case_files))

    start_time = time.time()
    plate_dict = plate_detection(case_files, args.case_path)
    plate_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    logging.info("Plate Detection Time: {}".format(plate_time_str))

    sys.exit(1)

    start_time = time.time()
    seg_dict = car_segmentation(case_files, args.case_path)
    seg_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    logging.info("Car Segmentation Time: {}".format(seg_time_str))
    logging.info("IMG Dataset:")
    for i in IMGDICT.keys():
        logging.info("key:{}, shape:{}".format(i, IMGDICT[i].shape))

    case_dict = case_division(seg_dict)

    cur_tim = time.time()
    color_name = color_detection(case_dict["colors"])
    seg_time_str = str(datetime.timedelta(seconds=int(time.time()-cur_tim)))
    logging.info("Color Prediction Time: {}".format(seg_time_str))

    cur_time = time.time()
    logo = logo_detection(case_dict["logos"])
    logo_time_str = str(datetime.timedelta(seconds=int(time.time()-cur_time)))
    logging.info("Logo Prediction Time: {}".format(logo_time_str))

    summary(seg_dict, logo, color_name, outputfolder=args.case_path, is_plot=True)

    logging.info("Total Time: {} s".format(time.time()-start_time))
    process = psutil.Process(os.getpid())
    MB = 1024*1024
    logging.info("Memory Usage: {} MB".format(process.memory_info().rss/MB))
