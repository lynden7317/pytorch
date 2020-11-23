import sys
import os
import math
import json
import numpy as np
import cv2
import torch
import time

import torchvision

#import coco_utils
#import coco_eval
#import utils
from dataloader import img_utils
from obj_detection.coco_utils import get_coco_api_from_dataset
from obj_detection.coco_eval import CocoEvaluator
from obj_detection import utils
from obj_detection import visualize


def train_one_epoch(model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    print_freq=10):
    """
    example:
      # construct an optimizer
      params = [p for p in model.parameters() if p.requires_grad]
      optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
      # and a learning rate scheduler
      lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
      num_epochs = 10
      for epoch in range(num_epochs):
          train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=10)
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def gen_coco_evaluator(model, data_loader):
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    return coco_evaluator

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def cal_iou(target, predict):
    iouDict = {}
    print(target['labels'], predict['masks'].shape, target['masks'].shape)
    labels = target['labels'].cpu().clone().numpy()
    masks = target['masks'].cpu().clone().numpy()
    h, w = masks.shape[1], masks.shape[2]
    print(labels, type(labels), predict['labels'], type(predict['labels']))
    summary_list = list(set(labels.tolist()+predict['labels'].tolist()))
    print("summary list: {}".format(summary_list))
    for k in summary_list:
        print(masks.shape)
        if k not in iouDict.keys():
            iouDict[k] = {'target': np.zeros((h, w)), 'predict': np.zeros((h, w)), 'iou': 0.0}

    for i, k in enumerate(labels):
        iouDict[k]['target'][np.where(masks[i,:,:]>=0.5)] = 1

    for i, k in enumerate(predict['labels']):
        if k in iouDict.keys():
            _mask = predict['masks'][i,:,:,0]
            iouDict[k]['predict'][np.where(_mask[:,:]>=0.5)] = 1

    for k in iouDict.keys():
        true_area = np.sum(iouDict[k]['target'])
        pred_area = np.sum(iouDict[k]['predict'])
        intersection = np.zeros((h, w))
        _union = iouDict[k]['target'] + iouDict[k]['predict']
        intersection[np.where(_union > 1.0)] = 1
        inter_area = np.sum(intersection)
        union = true_area + pred_area - inter_area
        iou = inter_area / union
        iouDict[k]['iou'] = iou
        iouDict[k]['true_area'] = true_area
        iouDict[k]['pred_area'] = true_area
        iouDict[k]['inter_area'] = inter_area
        #print("true area:{}, predict area:{}, intersection:{}".format(true_area, pred_area, inter_area))
        #print("iou: {}".format(iou))

    return iouDict


@torch.no_grad()
def evaluate(model, data_loader, device,
             class_names=[],
             score_threshold=0.7,
             is_plot=False,
             dump2LabelMe=False,
             plot_folder='./plotEval'):
    if is_plot:
        if not os.path.isdir(plot_folder):
            utils.mkdir(plot_folder)

    class_names = ['BG'] + class_names
    #n_threads = torch.get_num_threads()

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    batch = 0
    Stat_summary = {}
    IoUDict = {}
    for _c in class_names:
        Stat_summary[_c] = {'0.5':{'tp':0, 'fp':0, 'fn':0, 'precision':0.0, 'recall':0.0},
                            '0.6':{'tp':0, 'fp':0, 'fn':0, 'precision':0.0, 'recall':0.0},
                            '0.7':{'tp':0, 'fp':0, 'fn':0, 'precision':0.0, 'recall':0.0},
                            '0.8':{'tp':0, 'fp':0, 'fn':0, 'precision':0.0, 'recall':0.0},
                            '0.9':{'tp':0, 'fp':0, 'fn':0, 'precision':0.0, 'recall':0.0}}
        IoUDict[_c] = {'tp':[], 'fp':0, 'fn':0}

    for images, targets in metric_logger.log_every(data_loader, data_loader.batch_size, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        #sys.exit(1)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        for _i, t in enumerate(outputs):
            print(targets[_i].keys())
            if is_plot:
                img = images[_i].cpu().clone()
                img = torchvision.transforms.ToPILImage()(img)
                img = np.array(img)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _name = "img_b{}_{}.jpg".format(batch, _i)
                cv2.imwrite(os.path.join(plot_folder, _name), img_bgr)

            if len(t['labels']) == 0:
                print("no object is detected")
                continue

            _boxes = t['boxes'].cpu().clone().numpy()
            _labels = t['labels'].cpu().clone().numpy()
            _scores = t['scores'].cpu().clone().numpy()
            _masks = t['masks'].cpu().clone().numpy()
            _masks = _masks.transpose((0, 2, 3, 1))    # (N, C, H, W) --> (N, H, W, C)
            # ==== update results by score_threshold ==== #
            scores_pass = np.where(_scores > score_threshold)
            boxes = _boxes[scores_pass]
            labels = _labels[scores_pass]
            scores = _scores[scores_pass]
            masks = _masks[scores_pass]
            #masks = (_masks[scores_pass]).transpose(1, 2, 3, 0)[0]
            #print(scores_pass, scores_pass[0])
            print(boxes.shape, labels.shape, scores.shape, masks.shape)
            print(labels, targets[_i]['labels'])
            IoUs = cal_iou(targets[_i], {'labels':labels, 'masks':masks})
            for _k in IoUs.keys():
                print("{}, iou:{}".format(_k, IoUs[_k]['iou']))
                cname = class_names[_k]
                if IoUs[_k]['true_area'] > 0 and IoUs[_k]['pred_area'] > 0:
                    # tp
                    iou = IoUs[_k]['iou']
                    IoUDict[cname]['tp'].append(iou)
                elif IoUs[_k]['true_area'] == 0 and IoUs[_k]['pred_area'] > 0:
                    # fp
                    IoUDict[cname]['fp'] += 1
                elif IoUs[_k]['true_area'] > 0 and IoUs[_k]['pred_area'] == 0:
                    # fn
                    IoUDict[cname]['fn'] += 1
                else:
                    pass

            if is_plot:
                _name = os.path.join(plot_folder, "img_b{}_{}_pred.png".format(batch, _i))
                visualize.display_instances(img, boxes, masks, labels,
                                            class_names,
                                            is_save=[True, _name])

                _name = os.path.join(plot_folder, "img_b{}_{}_true.png".format(batch, _i))
                tlabels = targets[_i]['labels'].cpu().clone().numpy()
                tboxes = targets[_i]['boxes'].cpu().clone().numpy()
                tmasks = targets[_i]['masks'].cpu().clone().numpy()
                tmasks = np.expand_dims(tmasks, axis=3)
                print(tlabels.shape, tboxes.shape, tmasks.shape)
                visualize.display_instances(img, tboxes, tmasks, tlabels,
                                            class_names,
                                            is_save=[True, _name])

        batch += 1
        model_time = time.time() - model_time
        metric_logger.update(model_time=model_time)
        #if batch > 5:
        #    break
    print("IoUDict: {}".format(IoUDict))
    for cname in Stat_summary.keys():
        for _iou, _kiou in [(0.5, '0.5'), (0.6, '0.6'), (0.7, '0.7'), (0.8, '0.8'), (0.9, '0.9')]:
            tp_max = len(IoUDict[cname]['tp'])
            tp = len([_t for _t in IoUDict[cname]['tp'] if _t > _iou])
            fp = IoUDict[cname]['fp']
            fn = IoUDict[cname]['fn'] + (tp_max-tp)
            precision = float(tp)/float((tp+fp)) if (tp+fp) else 0.0
            recall = float(tp)/float((tp+fn)) if (tp+fn) else 0.0
            Stat_summary[cname][_kiou]['tp'] = tp
            Stat_summary[cname][_kiou]['fp'] = fp
            Stat_summary[cname][_kiou]['fn'] = fn
            Stat_summary[cname][_kiou]['precision'] = float("{:.4}".format(precision))
            Stat_summary[cname][_kiou]['recall'] = float("{:.4}".format(recall))

    print("Stat_summary: {}".format(Stat_summary))
    with open('evaluation.txt', 'w') as fid:
        fid.write(json.dumps(Stat_summary))

@torch.no_grad()
def evaluate_image(model, img_path, device,
                   class_names=[],
                   resize_img=[False, 512, 512],
                   padding=[True, 32],
                   score_threshold=0.7,
                   is_plot=False,
                   plot_folder='./plotEval'):
    """
    :param model:
    :param image: np.array
    :return:
    """
    if is_plot:
        if not os.path.isdir(plot_folder):
            utils.mkdir(plot_folder)

    class_names = ['BG'] + class_names

    img = cv2.imread(img_path)
    img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img_org = img_utils.gray_3_ch(img)
    #img_org = img_org.astype(np.uint8)

    img_pad_shape = img_org.shape
    img_tensor = img_org

    if padding[0]:
        padding = [(padding[1], padding[1]), (padding[1], padding[1]), (0, 0)]
        img_pad = np.pad(img_org, padding, mode='constant', constant_values=0)
        img_pad_shape = img_pad.shape
        img_tensor = img_pad
    if resize_img[0]:
        if padding[0]:
            in_img = img_pad
        else:
            in_img = img_org
        img_resize, resize_window, resize_scale, resize_padding, crop = img_utils.resize_image(in_img, min_dim=resize_img[1], max_dim=resize_img[2])
        img_tensor = img_resize

    #img = torchvision.transforms.ToPILImage()(img).convert('RGB')
    _img = torchvision.transforms.ToPILImage()(img_tensor)
    _img = torchvision.transforms.ToTensor()(_img)
    _img = torch.unsqueeze(_img, 0)
    _img = _img.to(device)

    model.eval()
    model_time = time.time()
    output = model(_img)[0]

    print(output.keys())
    #outputs = [{k: v.to(device) for k, v in t.items()} for t in output]
    #print(output)
    if len(output['labels']) == 0:
        print("no object is detected")
        return {"boxes": [], "labels": [], "scores": [], "masks": [], "class_names": class_names}

    _boxes = output['boxes'].cpu().clone().numpy()
    _labels = output['labels'].cpu().clone().numpy()
    _scores = output['scores'].cpu().clone().numpy()
    _masks = output['masks'].cpu().clone().numpy()
    _masks = _masks.transpose((0, 2, 3, 1))    # (N, C, H, W) --> (N, H, W, C)
    print("_masks.shape: ", _masks.shape)

    if resize_img[0]:
        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        #print(_boxes, "window", resize_window)
        wx1, wy1, wx2, wy2 = resize_window
        shift = np.array([wx1, wy1, wx1, wy1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        _scale = np.array([ww, wh, ww, wh])
        # Convert boxes to normalized coordinates on the window
        _boxes = np.divide(_boxes-shift, _scale)
        # Convert boxes to pixel coordinates on the original image
        _boxes = img_utils.denorm_boxes(_boxes, img_pad_shape[:2])
        #print(_boxes, img_pad.shape)
        full_masks = np.zeros((_masks.shape[0], img_pad_shape[0], img_pad_shape[1], 1))
        for _m in range(_masks.shape[0]):
            full_mask = img_utils.denorm_mask(_masks[_m, :, :, 0], resize_scale, resize_padding, img_pad_shape[:2])
            full_mask = np.expand_dims(full_mask, axis=2)
            full_masks[_m, :, :, :] = full_mask
        _masks = full_masks
    #print(_boxes.shape, _labels.shape, _scores.shape, _masks.shape)
    #sys.exit(1)

    # ==== update results by score_threshold ==== #
    scores_pass = np.where(_scores > score_threshold)
    boxes = _boxes[scores_pass]
    labels = _labels[scores_pass]
    scores = _scores[scores_pass]
    #masks = (_masks[scores_pass]).transpose(1, 2, 3, 0)[0]
    masks = _masks[scores_pass]
    #print(type(boxes), type(labels), type(scores), type(masks))
    #print(boxes.shape, labels.shape, scores.shape, masks.shape)
    result = {"boxes": boxes, "labels": labels, "scores": scores, "masks": masks, "class_names": class_names}

    model_time = time.time() - model_time
    print("Evaluation: model_time: {}".format(model_time))
    print(boxes.shape, labels.shape, scores.shape, masks.shape)

    if is_plot:
        #img = img[0].cpu().clone()
        #img = torchvision.transforms.ToPILImage()(img)
        #img = np.array(img)
        if resize_img[0]:
            img = in_img
        else:
            img = img_tensor

        dirpath = os.path.dirname(img_path)
        dirpath = os.path.basename(dirpath)
        plotname = os.path.basename(img_path).split(".")[0]
        _name = os.path.join(plot_folder, dirpath+"_"+plotname+"_pred.png")
        visualize.display_instances(img, boxes, masks, labels,
                                    class_names,
                                    is_display=False,
                                    is_save=[True, _name])

    return result

@torch.no_grad()
def evaluate_coco(model, data_loader, device, coco_evaluator=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if coco_evaluator is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    eval_values = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator, eval_values