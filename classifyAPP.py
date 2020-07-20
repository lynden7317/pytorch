import torch
import torchvision

from torch.utils.data import DataLoader

from dataloader import imgDataGen
from classification import engine
from classification import models
from classification import classifier

def app2():
    import torch.nn as nn
    from torchvision import models

    classIDs = {"0":0, "1":1, "2":2, "3":3}
    classes = 4
    pretrained_path = './training/resnet50_4_rotation_model.pth'

    # using MyCustomDataset
    dataset_val = imgDataGen.MyCustomDataset(datafolder="./data/rotation_val",
                                             pil_process=False,
                                             npy_process=True,
                                             height=224,
                                             width=224,
                                             classIDs=classIDs)
    dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False)

    p_model = models.resnet50()
    p_model.fc = nn.Sequential(
        nn.Linear(p_model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
        nn.Softmax()
    )

    p_model.load_state_dict(torch.load(pretrained_path))
    p_model.to(device)
    print("updated:", p_model)
    engine.evaluate(p_model, device, dataloader_val, classes_name=["0", "90", "180", "270"])

def app1():
    classIDs = {"4": 0, "5": 1}  # ["bank_cover", "bank_inner"]
    classes = 2
    pretrained_path = 'resnet50-19c8e357.pth' #'resnet18-5c106cde.pth'

    # using MyCustomDataset
    transforms = torchvision.transforms.Compose([torchvision.transforms.Pad(padding=32)])
    dataset_train = imgDataGen.MyCustomDataset(datafolder="./data/bank_cover_inner", transforms=transforms, height=256,
                                               width=256, \
                                               classIDs=classIDs)
    dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=True)
    dataset_val = imgDataGen.MyCustomDataset(datafolder="./data/bank_cover_inner_val", transforms=transforms,
                                             height=256, width=256, \
                                             classIDs=classIDs)
    dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False)
    dataloaders = {"train": dataloader_train, "val": dataloader_val}
    dataset_sizes = {"train": len(dataset_train), "val": len(dataset_val)}
    print(dataloaders, dataset_sizes)

    NNmodel, features = models.NNsModel(device, model_name='resnet50',
                                        classes=classes, pretrained_path=pretrained_path, extract_layers=[-2])

    #engine.extract_features(features[0], device, dataloaders['val'], classes_name=["bank_cover", "bank_inner"])

    classifier = classifier.xgboost_classifier(model_path="./classification/xgb_weights/xgb500.model", is_train=False)
    engine.feature2classifier(features[0], device, classifier, dataloader=dataloaders['val'], classes_name=["bank_cover", "bank_inner"])
    #engine.feature2classifier_image(features[0], device, classifier, img_path='test_bank.jpg', classes_name=["bank_cover", "bank_inner"])

    #NNmodel.load_state_dict(torch.load('best_model_wts'))
    #engine.evaluate(NNmodel, device, dataloaders['val'], classes_name=["bank_cover", "bank_inner"])
    #engine.evaluate_image(NNmodel, device, img_path='data/bank_cover_inner_val/4_76.jpg',
    #                      class_names=["bank_cover", "bank_inner"], is_plot=True, is_save=True)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    app2()


