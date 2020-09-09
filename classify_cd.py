import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataloader import imgDataGen
from classification import engine
from classification import models

import matplotlib.pyplot as plt

CLASSES = {'Alfa Romeo':0, 'Audi':1, 'BMW':2, 'Chevrolet':3, 'Citroen':4, 'Dacia':5, 'Daewoo':6, 'Dodge':7, 'Ferrari':8, 'Fiat':9, \
           'Ford':10, 'Honda':11, 'Hyundai':12, 'Jaguar':13, 'Jeep':14, 'Kia':15, 'Lada':16, 'Lancia':17, 'Land Rover':18, 'Lexus':19, \
           'Maserati':20, 'Mazda':21, 'Mercedes':22, 'Mitsubishi':23, 'Nissan':24, 'Opel':25, 'Peugeot':26, 'Porsche':27, 'Renault':28, 'Rover':29, \
           'Saab':30, 'Seat':31, 'Skoda':32, 'Subaru':33, 'Suzuki':34, 'Tata':35, 'Tesla':36, 'Toyota':37, 'Volkswagen':38, 'Volvo':39}

LCLASSES = ['Alfa Romeo', 'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Dacia', 'Daewoo', 'Dodge', 'Ferrari', 'Fiat', \
            'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Lada', 'Lancia', 'Land Rover', 'Lexus', \
            'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', \
            'Saab', 'Seat', 'Skoda', 'Subaru', 'Suzuki', 'Tata', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo']

import imgaug.augmenters as iaa
SEQ_AUG = iaa.SomeOf((1, 4), [
    iaa.Crop(percent=(0, 0.1)),
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
    iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    iaa.Flipud(0.5),
    iaa.Fliplr(0.5),
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

def app2():
    from imutils import paths

    ROOT = "./data/car_logo" #"./data/test_car_logo"  # "./data/damages"
    PRETRAINED = './weights/resnet50_10.pth'
    PAD = 8
    DIM = 128
    CLASSNUM = len(CLASSES)

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NNmodel, features = models.NNsModel(device, model_name='resnet50', classes=CLASSNUM)
    print(NNmodel)

    NNmodel.load_state_dict(torch.load(PRETRAINED))
    print("Load pretrained weight: {}".format(PRETRAINED))

    for _p in list(paths.list_images(ROOT)):
        #print("img: {}".format(_p))
        engine.evaluate_image(NNmodel, device, img_path=_p, height=DIM, width=DIM,
                              classes_name=LCLASSES, is_save=True)

def app():
    ROOT = "./data/car_logo" #"./data/damages"
    PRETRAINED = './classification/weights/resnet50-19c8e357.pth'
    PAD = 8
    DIM = 128
    EPOCHS = 100
    CLASSNUM = len(CLASSES)

    # Define transforms (1)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # Call the dataset
    dataset = imgDataGen.MyCustomDataset(root=ROOT,
                                         resize_img=[True, DIM, DIM],
                                         padding=[True, PAD],
                                         augmentation=SEQ_AUG,
                                         transforms=transforms,
                                         classIDs=CLASSES)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    dataloaders = {"train": dataloader}
    dataset_sizes = {"train": len(dataset)}

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NNmodel, features = models.NNsModel(device, model_name='resnet50', classes=CLASSNUM, pretrained_path=PRETRAINED)
    print(NNmodel)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    print("params: {}".format(NNmodel.parameters()))
    optimizer_ft = optim.SGD(NNmodel.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

    engine.train_model(device, NNmodel, criterion, optimizer_ft, exp_lr_scheduler,
                       dataloaders, dataset_sizes, phases=['train'], num_epochs=EPOCHS)

    """
    count = 1
    for inputs, targets in dataloader:
        print("input: ", type(inputs), len(inputs))
        for batch in range(len(inputs)):
            img = torchvision.transforms.ToPILImage()(inputs[batch])
            img = np.array(img)

            name = "./damages/" + str(count) + ".jpg"
            plt.imshow(img)
            plt.savefig(name)
            count += 1

        break
    """



if __name__ == '__main__':
    #root = "./data/car_logo"
    #classes = imgDataGen.genClasses(root)
    #print(classes)

    #app()

    app2()