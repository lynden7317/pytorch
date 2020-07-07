import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import sys
import math
import torch

import utils

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

    torch.save(model.state_dict(), "mrcnn_model_car_resnet.pth")

    return metric_logger


def MaskRCNN(backbone='resnet50',
             anchors=None,
             anchor_ratios=None,
             num_classes=2):
    """
    :param backbone:
    :param anchors:
    :param anchor_ratios:
    :param num_classes (int): number of output classes of the model (including the background).
    :return:
    """
    if anchors is None:
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
    else:
        if anchor_ratios is None:
            anchor_ratios = (0.5, 1.0, 2.0)
        anchor_generator = AnchorGenerator(sizes=(anchors,),
                                           aspect_ratios=(anchor_ratios,))

    if backbone == 'resnet50':
        return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif backbone == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
    else:
        pass


    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=14,
                                                         sampling_ratio=2)

    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     mask_roi_pool=mask_roi_pooler)

    return model
