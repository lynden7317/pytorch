import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn

from torchvision import models

import matplotlib.pyplot as plt

def imshow(inp, title=None, stop=False):
    """Imshow for Tensor."""
    print(type(inp))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    if stop:
        plt.show()
    else:
        plt.pause(1.0)

def img_eval_model(model, device, img_path, transforms=None, height=224, width=224, class_names=[]):
    import cv2

    was_training = model.training
    model.eval()

    with torch.no_grad():
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torchvision.transforms.ToPILImage()(img).convert('RGB')
        if transforms is not None:
            img = transforms(img)
        img = torchvision.transforms.Resize((height, width))(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torch.unsqueeze(img, 0)
        print(img.shape)

        input = img.to(device)
        output = model(input)
        _, pred = torch.max(output, 1)

        imshow(input.cpu().data[0], title=class_names[pred[0]], stop=True)

    model.train(mode=was_training)

def dataloader_eval_model(model, device, dataloader, num_images=6, class_names=[]):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                print(images_so_far)
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    imshow(inputs.cpu().data[images_so_far-1], stop=True)
                    return
                else:
                    imshow(inputs.cpu().data[images_so_far - 1])

        model.train(mode=was_training)

def NNsModel(device, model_name='resnet18', classes=2, pretrained_path=None, extract_layers=[]):
    features = []
    MODELS = {'vgg16': models.vgg16(), 'vgg16_bn': models.vgg16_bn(),\
              'resnet18': models.resnet18(), 'resnet34': models.resnet34(), 'resnet50': models.resnet50(),\
              'resnet101': models.resnet101(), 'resnet152': models.resnet152(),\
              'densenet121': models.densenet121(), 'densenet169': models.densenet169(),\
              'densenet201': models.densenet201(), 'mobilenetV2': models.mobilenet_v2()}

    model_ft = MODELS[model_name]
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

def train_model(device,
                model,
                criterion,
                optimizer,
                scheduler,
                dataloaders,
                dataset_sizes,
                num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if type(labels) in [tuple, list]:
                    labels = torch.from_numpy(np.asarray(labels))

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    torch.save(best_model_wts, "best_model_wts")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


if __name__ == '__main__':
    import imgDataGen

    import torchvision
    import torch.optim as optim

    from torch.utils.data import DataLoader
    from torch.optim import lr_scheduler

    torch.multiprocessing.freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classIDs = {"4":0, "5":1}
    classes = 2
    pretrained_path = 'resnet18-5c106cde.pth'

    # using ImageFolderDataLoader
    #dataloaders, dataset_sizes, class_names = imgDataGen.ImageFolderDataLoader("./data/hymenoptera_data", batch_size=10)

    # using MyCustomDataset
    transforms = torchvision.transforms.Compose([torchvision.transforms.Pad(padding=32)])
    dataset_train = imgDataGen.MyCustomDataset(datafolder="./data/bank_cover_inner", transforms=transforms, height=256, width=256,\
                                               classIDs=classIDs)
    dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=True)
    dataset_val = imgDataGen.MyCustomDataset(datafolder="./data/bank_cover_inner_val", transforms=transforms, height=256, width=256,\
                                             classIDs=classIDs)
    dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False)
    dataloaders = {"train": dataloader_train, "val":dataloader_val}
    dataset_sizes = {"train": len(dataset_train), "val": len(dataset_val)}
    print(dataloaders, dataset_sizes)

    NNmodel, features = NNsModel(device, classes=classes, pretrained_path=pretrained_path, extract_layers=[-2])
    #print(features)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    print("params: {}".format(NNmodel.parameters()))
    optimizer_ft = optim.SGD(NNmodel.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #model_ft = train_model(device, NNmodel, criterion, optimizer_ft, exp_lr_scheduler,
    #                       dataloaders, dataset_sizes, num_epochs=20)

    NNmodel.load_state_dict(torch.load('best_model_wts'))

    #dataloader_eval_model(NNmodel, device, dataloaders['val'], class_names=["bank_cover", "bank_inner"])
    img_eval_model(NNmodel, device, img_path='data/bank_cover_inner_val/4_76.jpg', class_names=["bank_cover", "bank_inner"])