import torch.nn as nn
import torchvision

from torchvision.models.detection.mask_rcnn import MaskRCNN as MRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

#((32,), (64,), (128,), (256,), (512,))
def MaskRCNN(backbone='default',
             anchors=(32, 64, 128, 256, 512),
             anchor_ratios=(0.5, 1.0, 2.0),
             num_classes=2):
    """
    :param backbone:
    :param anchors:
    :param anchor_ratios:
    :param num_classes (int): number of output classes of the model (including the background).
    :return:
    """
    if backbone == 'default':
        return torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=num_classes)
        #return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif backbone == 'resnet50':
        # anchor_sizes should be ((32,), (64,), (128,), (256,), (512,))
        backbone = resnet_fpn_backbone(backbone_name=backbone, pretrained=True)
    elif backbone == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
    else:
        pass

    anchor_sizes = tuple((s, ) for s in anchors)
    aspect_ratios = (anchor_ratios,)*len(anchor_sizes)
    #print(anchor_sizes, aspect_ratios)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    #print(anchor_generator.sizes, anchor_generator.aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=14,
                                                         sampling_ratio=2)

    model = MRCNN(backbone,
                  num_classes=num_classes,
                  rpn_anchor_generator=anchor_generator,
                  box_roi_pool=roi_pooler,
                  mask_roi_pool=mask_roi_pooler)

    return model
