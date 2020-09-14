"""
Cathay Car Damage Flow
"""
import os
import sys
import datetime
import time
import copy
import math
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
from classification import models

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

        return img_torch, pimg

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


def seg_criterion(tar, thresholds):
    _boxes = tar['boxes'].cpu().clone().numpy()
    _labels = tar['labels'].cpu().clone().numpy()
    _scores = tar['scores'].cpu().clone().numpy()
    _masks = tar['masks'].cpu().clone().numpy()
    _masks = _masks.transpose((0, 2, 3, 1))
    # ==== update results by score_threshold ==== #
    scores_pass = np.where(_scores > thresholds['score'])
    boxes = _boxes[scores_pass]
    labels = _labels[scores_pass]
    scores = _scores[scores_pass]
    masks = _masks[scores_pass]

    _nms_result = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), thresholds['nms'])
    nms_result = _nms_result.numpy()
    boxes = boxes[nms_result,:]
    labels = labels[nms_result]
    scores = scores[nms_result]
    masks = masks[nms_result,:,:,:]
    print("nms_result: {}".format(nms_result))

    return {"boxes":boxes, "labels":labels, "scores":scores, "masks":masks}

def major_car(cls_dict):
    # determine the major car segmentations
    pass

def car_mask_decode(image, result, classes, outputfolder):
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
    cv2.imwrite(path_org, img_bgr)
    labels = result["labels"]
    boxes = result["boxes"]
    masks = result["masks"][:,:,:,0]
    logging.debug("<car seg.> mask:{}, boxes:{}".format(masks.shape, boxes.shape))

    mask_img = np.zeros(img_bgr.shape)
    for _m in range(masks.shape[0]):
        #pos = np.where(masks[_m,:,:]>0.5)
        lab = labels[_m]
        x1,y1,x2,y2 = math.floor(boxes[_m, 0]), math.floor(boxes[_m, 1]), \
                      math.ceil(boxes[_m, 2]), math.ceil(boxes[_m, 3])
        w, h = x2-x1, y2-y1
        logging.debug("<car box> (x1,y1):{}, (w,h):{}".format((x1,y1), (w,h)))

        part_img = np.zeros(img_bgr.shape)
        pos_img = np.zeros(img_bgr.shape)
        _name = img_name+"_c_"+str(lab)+"_"+str(len(cls_list[lab]))+".jpg"
        for c in range(3):
            mask_img[:,:,c] = np.where(masks[_m,:,:]>0.5, img_bgr[:,:,c], mask_img[:,:,c])
            part_img[:,:,c] = np.where(masks[_m,:,:]>0.5, img_bgr[:,:,c], 0)
            pos_img[:,:,c] = np.where(masks[_m,:,:]>0.5, 1, 0)

        path = os.path.join(outputfolder, _name)
        part_img = part_img[y1:y2,x1:x2,:]
        pos_img = pos_img[y1:y2,x1:x2,:]
        cv2.imwrite(path, part_img)
        cls_list[lab].append((path_org, path, (x1,y1), (w,h), part_img, pos_img))

    #logging.debug("predict list:{}".format(cls_list))

    _name = img_name+"_c.jpg"
    cv2.imwrite(os.path.join(outputfolder, _name), mask_img)

    for i, c in enumerate(classes):
        cls_dict[c] = cls_list[i]

    major_car(cls_dict)

    #logging.debug("predict dict:{}".format(cls_dict))
    return cls_dict

def damage_mask_decode(image, result, classes, outputfolder):
    cls_list = [[] for c in classes]
    path = result["path"]
    img_name = os.path.basename(path).split(".")[0]
    img = image.cpu().clone()
    img = torchvision.transforms.ToPILImage()(img)
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _name = img_name + ".jpg"
    path_org = os.path.join(outputfolder, _name)

    labels = result["labels"]
    boxes = result["boxes"]
    masks = result["masks"][:, :, :, 0]
    logging.debug("<damage seg.> mask:{}, boxes:{}".format(masks.shape, boxes.shape))

    mask_img = np.zeros(img_bgr.shape)
    for _m in range(masks.shape[0]):
        lab = labels[_m]
        x1, y1, x2, y2 = math.floor(boxes[_m, 0]), math.floor(boxes[_m, 1]), \
                         math.ceil(boxes[_m, 2]), math.ceil(boxes[_m, 3])
        w, h = x2 - x1, y2 - y1
        logging.debug("<damage box> (x1,y1):{}, (w,h):{}".format((x1, y1), (w, h)))

        part_img = np.zeros(img_bgr.shape)
        _name = img_name + "_d_" + str(lab) + "_" + str(len(cls_list[lab])) + ".jpg"
        for c in range(3):
            mask_img[:,:,c] = np.where(masks[_m,:,:] > 0.5, img_bgr[:,:,c], mask_img[:,:,c])
            part_img[:,:,c] = np.where(masks[_m,:,:] > 0.5, img_bgr[:,:,c], 0)

        path = os.path.join(outputfolder, _name)
        part_img = part_img[y1:y2, x1:x2, :]
        cv2.imwrite(path, part_img)
        cls_list[lab].append((path_org, path, (x1, y1), (w, h)))

    _name = img_name + "_d.jpg"
    cv2.imwrite(os.path.join(outputfolder, _name), mask_img)

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
    CLASS = ['DS', 'DD', 'DC', 'DW', 'DH'] #['D']

    mrcnn = mask_rcnn.MaskRCNN(backbone=BACKBONE,
                               anchor_ratios=(0.33, 0.5, 1, 2, 3),
                               num_classes=1 + len(CLASS))

    mrcnn.load_state_dict(torch.load(MODEL_PATH))
    mrcnn.to(device)
    mrcnn.eval()

    return mrcnn, CLASS

@torch.no_grad()
def car_segmentation(cases, folder):
    DIM = 1024
    PAD = 32
    BATCHSIZE = 16
    CAR_SCORETHRESHOLD = 0.7
    CAR_NMSTHRESHOLD = 0.3
    DAMAGE_SCORETHRESHOLD = 0.7
    DAMAGE_NMSTHRESHOLD = 0.3

    TMPFOLDER = os.path.join(folder, 'tmp')
    if not os.path.isdir(TMPFOLDER):
        try:
            os.makedirs(TMPFOLDER)
        except:
            logging.error("cannot create folder at path:{}".format(TMPFOLDER))
            TMPFOLDER = folder


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
    car_class_dict = dict((k, []) for k in car_class_names)
    for images, targets in metric_logger.log_every(dbloader, dbloader.batch_size, header):
        images = list(img.to(device) for img in images)
        torch.cuda.synchronize()
        c_outputs = c_mrcnn(images)   # car segmentation
        d_outputs = d_mrcnn(images)   # damage segmentation
        c_outputs = [{k: v.to(device) for k, v in t.items()} for t in c_outputs]
        d_outputs = [{k: v.to(device) for k, v in t.items()} for t in d_outputs]
        print(len(c_outputs), len(d_outputs))
        for _i, c in enumerate(c_outputs):
            logging.debug("car seg. target keys:{}".format(targets[_i].keys()))
            d = d_outputs[_i]
            img_idx = targets[_i]['idx'].cpu().clone().numpy()
            img_path = imgset.imgs[img_idx]

            if len(c['labels']) > 0:
                c_results = seg_criterion(c, {'score':CAR_SCORETHRESHOLD, 'nms':CAR_NMSTHRESHOLD})
                c_results.update({"path": img_path})
                mask_dict = car_mask_decode(images[_i], c_results, car_class_names, TMPFOLDER)
                for _k in mask_dict.keys():
                    car_class_dict[_k] += mask_dict[_k]

            if len(d['labels']) > 0:
                d_results = seg_criterion(d, {'score':DAMAGE_SCORETHRESHOLD, 'nms':DAMAGE_NMSTHRESHOLD})
                d_results.update({"path": img_path})
                damage_mask_decode(images[_i], d_results, damage_class_names, TMPFOLDER)

    #logging.debug("case seg. results:{}".format(class_dict))
    return car_class_dict

def case_division(seg_dict):
    case_dict = {"plates":[], "logos":[], "colors":[], "damages":[]}

    for c in seg_dict.keys():
        if c in ["CC", "CP", "CL"]:
            continue
        else:
            case_dict["colors"] += copy.deepcopy(seg_dict[c])

    case_dict["logos"] = copy.deepcopy(seg_dict["CL"])

    return case_dict


def plate_detection(case_list):
    pass

def color_detection(case_list):
    # (path_org, path, (x1,y1), (w,h), part_img, pos_img)
    # received color seq: BGR
    #logging.debug("colors: {}".format(case_list))
    grid_size = 10
    colors = {}
    for c in case_list:
        logging.debug("path:{}, part_img:{}, pos_img:{}".format(c[1], c[4].shape, c[5].shape))
        x = np.linspace(0, c[4].shape[1], grid_size+1, dtype=np.int32)
        y = np.linspace(0, c[4].shape[0], grid_size+1, dtype=np.int32)
        #print(x, y)
        pos_mask = c[5]
        mask_img = c[4]
        for gx in range(grid_size):
            x1, x2 = x[gx], x[gx + 1]
            for gy in range(grid_size):
                y1, y2 = y[gy], y[gy+1]
                avg = np.sum(pos_mask[y1:y2, x1:x2, 0], dtype=np.int32)
                color_bgr = np.sum(np.sum(mask_img[y1:y2, x1:x2, :], axis=0), axis=0)
                if avg > 0:
                    avg = float(1.0/float(avg))
                    color_bgr_avg = np.array(color_bgr*avg, dtype=np.int32)
                    cname = cathay_utils.closest_colour((color_bgr_avg[2],color_bgr_avg[1],color_bgr_avg[0]))
                    if cname in colors.keys():
                        colors[cname] += 1
                    else:
                        colors[cname] = 1
                    #print(avg, color_bgr_avg, cname)

    max_color = sorted(colors, key=lambda k: colors[k])
    colorName = max_color[-1] + "_" + max_color[-2]
    logging.debug("color name: {}".format(colorName))


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

    cases = [i[0] for i in case_list]
    logging.debug("<logo detection> cases:{}".format(cases))

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
    for images, targets in metric_logger.log_every(dbloader, dbloader.batch_size, header):
        images = images.to(device)
        torch.cuda.synchronize()
        outputs = resnet(images)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().clone().numpy().tolist()
        results.append(max(set(preds), key=lambda x:preds.count(x)))
        #print(preds, type(preds))

    max_pred_logo = CLASSES[max(set(results), key=lambda x:results.count(x))]
    logging.debug("<logo> predicted logo:{}".format(max_pred_logo))
    return max_pred_logo



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
            fpath = cathay_utils.path_join(folder, f)
            case_files.append(fpath)

    logging.debug("files in case:{}".format(case_files))

    start_time = time.time()
    seg_dict = car_segmentation(case_files, args.case_path)
    seg_time = time.time()
    seg_time_str = str(datetime.timedelta(seconds=int(seg_time-start_time)))
    logging.info("Car Segmentation Time: {}".format(seg_time_str))

    case_dict = case_division(seg_dict)

    plate_detection(case_dict["plates"])

    #color_detection(case_dict["colors"])
    seg_time_str = str(datetime.timedelta(seconds=int(time.time()-seg_time)))
    logging.info("Color Prediction Time: {}".format(seg_time_str))

    #logo_detection(case_dict["logos"])

    logging.info("Total Time:{}".format(time.time()-start_time))
