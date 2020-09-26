from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models.detection.mask_rcnn import MaskRCNN as MRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

#((32,), (64,), (128,), (256,), (512,))
def MaskRCNN(backbone='default',
             anchors=(32, 64, 128, 256, 512),
             anchor_ratios=(0.5, 1.0, 2.0),
             num_classes=2,
             pretrained=True):
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
    elif backbone in ['resnet50', 'resnet101']:
        # anchor_sizes should be ((32,), (64,), (128,), (256,), (512,))
        # backbone = resnet_fpn_backbone(backbone_name=backbone, pretrained=pretrained)
        backboneNet = resnet_fpn_backbone(backbone, pretrained=pretrained)
    elif backbone == 'mobilenet_v2':
        backboneNet = torchvision.models.mobilenet_v2(pretrained=pretrained).features
        backboneNet.out_channels = 1280
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

    # output_size=14
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=28,
                                                         sampling_ratio=2)
    mask_head = None
    out_chs = backboneNet.out_channels
    mask_dilation = 1
    mask_head = MaskRCNNHeads(out_chs, mask_dilation)

    model = MRCNN(backboneNet,
                  num_classes=num_classes,
                  rpn_anchor_generator=anchor_generator,
                  box_roi_pool=roi_pooler,
                  mask_roi_pool=mask_roi_pooler,
                  mask_head=mask_head)

    return model

class MaskRCNNHeads(nn.Module):
    def __init__(self, in_channels, dilation):
        super(MaskRCNNHeads, self).__init__()

        next_feature = in_channels
        self.mask_fcn1 = nn.Conv2d(next_feature, 256, kernel_size=3,
                                   stride=1, padding=dilation, dilation=dilation)
        self.mask_fcn2 = nn.Conv2d(256, 256, kernel_size=3,
                                   stride=1, padding=dilation, dilation=dilation)
        self.mask_fcn3 = nn.Conv2d(256, 256, kernel_size=3,
                                   stride=1, padding=dilation, dilation=dilation)
        self.mask_fcn4 = nn.Conv2d(256, 256, kernel_size=3,
                                   stride=1, padding=dilation, dilation=dilation)
        self.mask_fcn5_t = nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.mask_fcn6_t = nn.ConvTranspose2d(256, 256, 2, 2, 0)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        #print("before:{}".format(x.shape))
        branch1 = F.relu(self.mask_fcn1(x))
        x = F.max_pool2d(self.mask_fcn1(x), (2,2))
        branch2 = F.relu(self.mask_fcn2(x))
        x = F.max_pool2d(self.mask_fcn2(x), (2,2))
        #print(x.shape)
        x = self.mask_fcn5_t(x)
        #print(x.shape)
        x = F.relu(self.mask_fcn3(x+branch2))
        x = self.mask_fcn6_t(x)
        #print(x.shape)
        x = F.relu(self.mask_fcn4(x+branch1))
        #print("after:{}".format(x.shape))
        return x


"""
CLASSES = 15
mrcnn = MaskRCNN(backbone='resnet50',
                 anchor_ratios=(0.33, 0.5, 1, 2, 3),
                 num_classes=1 + CLASSES)

print(mrcnn)
print(list(mrcnn.children()))
print(mrcnn.roi_heads.mask_head)
"""