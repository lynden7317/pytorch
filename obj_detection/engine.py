import sys
import os
import math
import numpy as np
import cv2
import torch
import time

import torchvision

#import coco_utils
#import coco_eval
#import utils
from obj_detection.coco_utils import get_coco_api_from_dataset
from obj_detection.coco_eval import CocoEvaluator
from obj_detection import utils
from obj_detection import visualize


def train_one_epoch(model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    print_freq):
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

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

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

        if lr_scheduler is not None:
            lr_scheduler.step()

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

@torch.no_grad()
def evaluate(model, data_loader, device, class_names=[],
             score_threshold=0.7,
             is_plot=False,
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
    for images, targets in metric_logger.log_every(data_loader, data_loader.batch_size, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        for _i, t in enumerate(outputs):
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
            #_masks = t['masks'].cpu().clone().numpy().transpose(1, 2, 3, 0)[0]
            # ==== update results by score_threshold ==== #
            scores_pass = np.where(_scores > score_threshold)
            boxes = _boxes[scores_pass]
            labels = _labels[scores_pass]
            scores = _scores[scores_pass]
            masks = (_masks[scores_pass]).transpose(1, 2, 3, 0)[0]
            #print(scores_pass, scores_pass[0])
            #print(boxes.shape, labels.shape, scores.shape, masks.shape)
            if is_plot:
                _name = os.path.join(plot_folder, "img_b{}_{}_pred.png".format(batch, _i))
                visualize.display_instances(img, boxes, masks, labels,
                                            class_names,
                                            is_save=[True, _name])

        batch += 1
        model_time = time.time() - model_time
        metric_logger.update(model_time=model_time)
        if batch > 5:
            break


@torch.no_grad()
def evaluate_image(model, img_path, device, class_names=[],
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torchvision.transforms.ToPILImage()(img).convert('RGB')
    img = torchvision.transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    model.eval()
    model_time = time.time()
    output = model(img)[0]

    print(output)
    #outputs = [{k: v.to(device) for k, v in t.items()} for t in output]
    #print(output)
    if len(output['labels']) == 0:
        print("no object is detected")
        return {"boxes": [], "labels": [], "scores": [], "masks": [], "class_names": class_names}

    _boxes = output['boxes'].cpu().clone().numpy()
    _labels = output['labels'].cpu().clone().numpy()
    _scores = output['scores'].cpu().clone().numpy()
    _masks = output['masks'].cpu().clone().numpy()
    # ==== update results by score_threshold ==== #
    scores_pass = np.where(_scores > score_threshold)
    boxes = _boxes[scores_pass]
    labels = _labels[scores_pass]
    scores = _scores[scores_pass]
    masks = (_masks[scores_pass]).transpose(1, 2, 3, 0)[0]
    print(type(boxes), type(labels), type(scores), type(masks))
    result = {"boxes": boxes, "labels": labels, "scores": scores, "masks": masks, "class_names": class_names}

    model_time = time.time() - model_time
    print("Evaluation: model_time: {}".format(model_time))
    print(boxes.shape, labels.shape, scores.shape, masks.shape)

    if is_plot:
        img = img[0].cpu().clone()
        img = torchvision.transforms.ToPILImage()(img)
        img = np.array(img)
        _name = os.path.join(plot_folder, os.path.basename(img_path).split(".")[0]+"_pred.png")
        visualize.display_instances(img, boxes, masks, labels,
                                    class_names,
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