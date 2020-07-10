import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator


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
