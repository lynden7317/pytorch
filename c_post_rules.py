import os
import sys
import cv2
import copy
import numpy as np
import torch
import logging

import torchvision

import cathay_cd
import cathay_utils

from classification import models


LCLASSES = ['B', 'F', 'R', 'L', 'BR', 'BL', 'FR', 'FL']
CAR_EXCLUDE = {'B':['F', 'FR', 'FL'], 'F':['B', 'BR', 'BL'], \
               'R':['L', 'FL', 'BL'], 'L':['R', 'FR', 'BR'], \
               'BR':['F', 'L', 'FL', 'BL', 'FR'], 'BL':['F', 'R', 'FR', 'BR', 'FL'], \
               'FR':['B', 'L', 'FL', 'BL', 'BR'], 'FL':['B', 'R', 'BR', 'BL', 'FR']}
CAR_CHECK = {'F': ['CAF', 'CBF', 'CLF', 'CG', 'CWF'],
             'FR': ['CAF', 'CBF', 'CLF', 'CG', 'CWF'],
             'FL': ['CAF', 'CBF', 'CLF', 'CG', 'CWF'],
             'B': ['CAB', 'CBB', 'CLB', 'CWB'],
             'BR': ['CAB', 'CBB', 'CLB', 'CWB'],
             'BL': ['CAB', 'CBB', 'CLB', 'CWB'],
             'L': [],
             'R': []
             }

# ==== CAM: Class Activation Map ==== #
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    print(cam.shape, img.shape)
    #print(cam)
    tar_img = np.zeros(cam.shape)
    print(tar_img.shape)
    for c in range(3):
        tar_img[:,:,c] = np.where(cam[:,:,c] > 0.7, img[:,:,c], 0)

    cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    cv2.imwrite("cam_1.jpg", (255 * tar_img))

# ===== CAM done ==== #

def img2torch(img, transforms=None, height=224, width=224):
    img = torchvision.transforms.ToPILImage()(img).convert('RGB')
    if transforms is not None:
        img = transforms(img)
    img = torchvision.transforms.Resize((height, width))(img)
    img = torchvision.transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)

    return img

def position_merge(img_dict, cls_dict):
    print(cls_dict)
    for _k in cls_dict.keys():
        if _k in ['CAF', 'CAB', 'CBF', 'CBB', 'CDFR', 'CDFL', 'CDBR', 'CDBL', \
                  'CFFR', 'CFFL', 'CFBR', 'CFBL', 'CS', 'CG']:
            updated_pos = []
            pos = cls_dict[_k]
            if len(pos) > 1:
                # sorting by area
                sorted_pos = []
                for i, v in enumerate(pos):
                    sorted_pos.append((i, len(v[4][0])))
                sorted_pos = sorted(sorted_pos, key=lambda tup: tup[1], reverse=True)
                print(sorted_pos)

                org_shape = img_dict[pos[0][0]].shape
                print("org_shape:{}".format(org_shape))

                target = pos[sorted_pos[0][0]]
                merge_list = [sorted_pos[0][0]]
                for i in range(len(sorted_pos)-1):
                    compare = pos[sorted_pos[i+1][0]]
                    print("target: {}, compare: {}".format(target, compare))
                    merge = cathay_utils.cal_iou(target, compare, org_shape)
                    if merge:
                        merge_list.append(sorted_pos[i+1][0])

                print(merge_list)

                t_pos = np.zeros((org_shape[0], org_shape[1]))
                t_cols, t_rows = target[4]
                t_pos[t_cols, t_rows] = 1
                for i in merge_list:
                    m_pos = np.zeros((org_shape[0], org_shape[1]))
                    m_cols, m_rows = pos[i][4]
                    m_pos[m_cols, m_rows] = 1
                    t_pos = t_pos + m_pos

                # ==== update pos ==== #
                nt_cols, nt_rows = np.where(t_pos > 0.5)
                x, y = np.min(nt_rows), np.min(nt_cols)
                w, h = np.max(nt_rows)-x, np.max(nt_cols)-y
                updated_pos.append((target[0], target[1], (x,y), (w,h), (nt_cols,nt_rows), target[5]))
                for i, v in enumerate(pos):
                    if i in merge_list:
                        continue
                    updated_pos.append(copy.deepcopy(v))
                print(updated_pos)
                cls_dict[_k] = updated_pos


def side_check(image):
    MODEL_PATH = "./weights/car_side/20201210/best_resnet50.pth"
    MODEL = 'resnet50'
    PAD = 8
    DIM = 256

    print(image.shape, type(image))
    img = image.cpu().clone()
    img = torchvision.transforms.ToPILImage()(img).convert('RGB')
    img = np.array(img)

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    imgset = cathay_cd.ImgSet([img],
                              resize_img=[True, DIM, DIM],
                              padding=[True, PAD],
                              transforms=transforms)

    dbloader = torch.utils.data.DataLoader(imgset, batch_size=2, shuffle=False)
    #print(imgset[0])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    logging.info("Execute <CAR RULES CHECK>")

    resnet, features = models.NNsModel(device, model_name=MODEL, classes=len(LCLASSES))
    resnet.load_state_dict(torch.load(MODEL_PATH))
    logging.info("<logo detection> load pretrained weight: {}".format(MODEL_PATH))

    resnet.eval()
    for inp in dbloader:
        input = inp.to(device)

        img = input[0].mul(255).byte()
        img = img.cpu().numpy().transpose((1, 2, 0))
        # for opencv
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./tttt.jpg", img_bgr)
        print(img.shape, type(img))

        torch.cuda.synchronize()
        outputs = resnet(input)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().clone().numpy().tolist()
        print("predict: {}".format(preds))
        break

    return LCLASSES[preds[0]]

"""
        #logging.info("<logo detection> CAM detection")
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.

        target_index = None
        img_cam = input[0].byte()
        img_cam = img_cam.cpu().numpy().transpose((1, 2, 0))
        img_cam = img_cam.astype(np.float32)
        print("cam: {}".format(img_cam.shape))
        cam_input = preprocess_image(img_cam)
        mask = grad_cam(cam_input, target_index)
        show_cam_on_image(img_bgr, mask)
"""


def remove_wrongs(side, cls_dict):
    # condition:
    # 1. use CAR_EXCLUDE to find the CAR_CHECK
    # 2. check the existed overlap area, > 0.7 remove it
    car_ex = CAR_EXCLUDE[side]
    car_checks = []  # filter the wrong parts
    for i in car_ex:
        car_checks += CAR_CHECK[i]
    car_checks = list(set(car_checks))
    print(cls_dict)
    print(car_checks)
    for ck in car_checks:       # name
        if len(cls_dict[ck]) > 0:
            checks = cls_dict[ck]
            updated_list = []
            for _ck in checks:  # item
                xa, ya = _ck[2]
                wa, ha = _ck[3]
                bbox1 = [xa, ya, xa+wa, ya+ha]
                is_wrong = False
                for _c in cls_dict.keys():  # name
                    if _c in ['CTA', 'CP']:
                        continue
                    if _c == ck:
                        continue
                    for cc in cls_dict[_c]:
                        xb, yb = cc[2]
                        wb, hb = cc[3]
                        bbox2 = [xb, yb, xb+wb, yb+hb]
                        iou = cathay_utils.bbox_iou(bbox1, bbox2)
                        print(ck, _c, iou)
                        if iou > 0.7:   # wrong item, remove it
                            is_wrong = True
                            break
                if is_wrong:
                    pass
                else:
                    updated_list.append(_ck)

            cls_dict[ck] = updated_list

    print(cls_dict)
    #sys.exit(1)

def correct_wrongs(side, cls_dict):
    pass