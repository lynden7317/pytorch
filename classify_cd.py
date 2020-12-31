import sys
import os
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataloader import imgDataGen
from classification import engine
from classification import models
from imutils import paths

import matplotlib.pyplot as plt

import cathay_utils


CLASSES = {'Alfa Romeo':0, 'Audi':1, 'BMW':2, 'Chevrolet':3, 'Citroen':4, 'Dacia':5, 'Daewoo':6, 'Dodge':7, 'Ferrari':8, 'Fiat':9, \
           'Ford':10, 'Honda':11, 'Hyundai':12, 'Jaguar':13, 'Jeep':14, 'Kia':15, 'Lada':16, 'Lancia':17, 'Land Rover':18, 'Lexus':19, \
           'Maserati':20, 'Mazda':21, 'Mercedes':22, 'Mitsubishi':23, 'Nissan':24, 'Opel':25, 'Peugeot':26, 'Porsche':27, 'Renault':28, 'Rover':29, \
           'Saab':30, 'Seat':31, 'Skoda':32, 'Subaru':33, 'Suzuki':34, 'Tata':35, 'Tesla':36, 'Toyota':37, 'Volkswagen':38, 'Volvo':39, 'Luxgen':40, \
           'Mini':41, 'Smart':42, 'Infiniti':43}

LCLASSES = ['Alfa Romeo', 'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Dacia', 'Daewoo', 'Dodge', 'Ferrari', 'Fiat', \
            'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Lada', 'Lancia', 'Land Rover', 'Lexus', \
            'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', \
            'Saab', 'Seat', 'Skoda', 'Subaru', 'Suzuki', 'Tata', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo', 'Luxgen', \
            'Mini', 'Smart', 'Infiniti']

"""
CLASSES = {'B':0, 'F':1, 'R':2, 'L':3, 'BR':4, 'BL':5, 'FR':6, 'FL':7}
LCLASSES = ['B', 'F', 'R', 'L', 'BR', 'BL', 'FR', 'FL']
"""

import imgaug.augmenters as iaa
SEQ_AUG = iaa.SomeOf((1, 4), [
    iaa.Crop(percent=(0, 0.1)),
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
    iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    #iaa.Flipud(0.5),
    #iaa.Fliplr(0.5),
    iaa.AddToHue((-50, 50)),
    iaa.WithColorspace(
        to_colorspace="HSV",
        from_colorspace="RGB",
        children=iaa.WithChannels(
            0, iaa.Add((0, 50)))
    ),
    iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
    iaa.MultiplyHueAndSaturation((0.5, 1.5)),
    iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((50, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]
    ),
    iaa.Grayscale(alpha=(0.0, 1.0)),
])

NAME_SPLIT_FUNC = lambda x:(re.compile(r'[a-zA-Z ]+')).findall(x)[0]

def app(apptype='train'):
    ROOT = 'D:/Cathay_DB/Cathay_Logo_Classify/CTA_test/CTADIR_20201211_2'
    #"./data/Cathay_Logo_Classify/20201215" #"./data/damages"
    PRETRAINED = './weights/car_logo/20201215/best_resnet50.pth'
    #'./weights/Classify/Side_20201209/best_resnet50.pth'
    #'./classification/weights/resnet50-19c8e357.pth'

    # ==== parameters ==== #
    BATCHSIZE = 10
    if apptype == 'eval':
        AUG = None
        SHUFFLE = False
    else:
        AUG = SEQ_AUG
        SHUFFLE = True
    DIM = 256
    PAD = 8
    EPOCHS = 100
    CLASSNUM = len(LCLASSES)
    # =================== #

    if apptype == 'eval':
        torch.multiprocessing.freeze_support()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        NNmodel, features = models.NNsModel(device, model_name='resnet50', classes=CLASSNUM)
        print(NNmodel)

        NNmodel.load_state_dict(torch.load(PRETRAINED))
        print("Load pretrained weight: {}".format(PRETRAINED))

        TP, TF = 0, 0
        TotalCount = 0
        TAGTTRUE = False
        for _p in list(paths.list_images(ROOT)):
            print("img: {}".format(_p))
            #pred_labs = engine.evaluate_image(NNmodel, device, img_path=_p, topk=2, height=DIM, width=DIM,
            #                                  classes_name=LCLASSES, is_save=True, save_folder='side_clf_eval')

            pred_labs = engine.evaluate_image(NNmodel, device, img_path=_p, topk=1, height=DIM, width=DIM,
                                              classes_name=LCLASSES, is_save=True, save_folder='logo_clf_eval', tagTrue=TAGTTRUE)

            if TAGTTRUE:
                img_name = (os.path.basename(_p)).split('.jpg')[0]
                true_lab = NAME_SPLIT_FUNC(img_name)
                if true_lab in pred_labs:
                    TP += 1
                else:
                    TF += 1

            TotalCount += 1

        if TAGTTRUE:
            ACCURACY = TP/(TP+TF)
            print("TP={}, TF={}, Accuracy: {}".format(TP, TF, ACCURACY))
            print("Total count={}".format(TotalCount))
        else:
            print("Total count={}".format(TotalCount))

        print("evaluation done")
        return

    # Define transforms (1)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # Call the dataset
    dataset = imgDataGen.MyCustomDataset(root=ROOT,
                                         resize_img=[True, DIM, DIM],
                                         padding=[True, PAD],
                                         augmentation=AUG,
                                         transforms=transforms,
                                         classIDs=CLASSES)

    dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=SHUFFLE)
    dataloaders = {"train": dataloader}
    dataset_sizes = {"train": len(dataset)}

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NNmodel, features = models.NNsModel(device, model_name='resnet50', classes=CLASSNUM, pretrained_path=PRETRAINED)
    #NNmodel.load_state_dict(torch.load(PRETRAINED))
    print(NNmodel)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    print("params: {}".format(NNmodel.parameters()))
    optimizer_ft = optim.SGD(NNmodel.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.94)

    engine.train_model(device, NNmodel, criterion, optimizer_ft, exp_lr_scheduler,
                       dataloaders, dataset_sizes, phases=['train'], num_epochs=EPOCHS,
                       outputfolder="./logo_cls_training_1215")


if __name__ == '__main__':
    args = cathay_utils.parse_commands()

    #root = "./data/car_logo"
    #classes = imgDataGen.genClasses(root)
    #print(classes)

    app(apptype=args.mode)