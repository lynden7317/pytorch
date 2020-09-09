import torch
import torch.nn as nn

from torchvision import models

def load_partial_weights(pretrain, model):
    pretrain_dict = torch.load(pretrain)
    model_dict = model.state_dict()
    load_dict = {}
    for k, v in pretrain_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                # print(k, v.shape, model_dict[k].shape)
                load_dict.update({k: v})

    model_dict.update(load_dict)
    model.load_state_dict(model_dict)

    return model

def NNsModel(device, model_name='resnet18', classes=2, pretrained_path=None, extract_layers=[]):
    features = []
    MODELS = {'vgg16': models.vgg16(), 'vgg16_bn': models.vgg16_bn(),\
              'resnet18': models.resnet18(), 'resnet34': models.resnet34(), 'resnet50': models.resnet50(),\
              'resnet101': models.resnet101(), 'resnet152': models.resnet152(),\
              'densenet121': models.densenet121(), 'densenet169': models.densenet169(),\
              'densenet201': models.densenet201(), 'mobilenetV2': models.mobilenet_v2()}

    model_ft = MODELS[model_name]
    model_ft.name = model_name
    if pretrained_path is not None:
        model_ft.load_state_dict(torch.load(pretrained_path))
    num_ftrs = model_ft.fc.in_features

    if len(extract_layers) > 0:
        for _l in extract_layers:
            features.append(nn.Sequential(*list(model_ft.children())[:_l]))

    print("num_ftrs: {}".format(num_ftrs))
    #print("features: {}".format(features))

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, classes)

    model_ft = model_ft.to(device)

    return model_ft, features