import os
import errno
import time
import copy
import cv2
import torch
import numpy as np

import torch.nn as nn
import torchvision

from classification import visualize

def img2torch(img_path, transforms=None, height=224, width=224):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torchvision.transforms.ToPILImage()(img).convert('RGB')
    if transforms is not None:
        img = transforms(img)
    img = torchvision.transforms.Resize((height, width))(img)
    img = torchvision.transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)

    return img

@torch.no_grad()
def evaluate_image(model, device,
                   img_path,
                   transforms=None,
                   height=224, width=224,
                   classes_name=[],
                   is_plot=False,
                   is_save=False,
                   plot_folder='./plotEval'):

    if is_save:
        if not os.path.isdir(plot_folder):
            try:
                os.makedirs(plot_folder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    was_training = model.training
    model.eval()

    img = img2torch(img_path, transforms, height, width)

    input = img.to(device)
    output = model(input)
    _, pred = torch.max(output, 1)

    if is_plot:
        if is_save:
            _name = os.path.join(plot_folder, os.path.basename(img_path).split(".")[0]+"_pred.png")
            visualize.imshow(input.cpu().data[0], title=classes_name[pred[0]], is_display=True, is_save=[True, _name])
        else:
            visualize.imshow(input.cpu().data[0], title=classes_name[pred[0]], is_display=True)
    else:
        if is_save:
            _name = os.path.join(plot_folder, os.path.basename(img_path).split(".")[0]+"_pred.png")
            visualize.imshow(input.cpu().data[0], title=classes_name[pred[0]], is_display=False, is_save=[True, _name])

    model.train(mode=was_training)

@torch.no_grad()
def evaluate(model, device, dataloader, classes_name=[]):
    was_training = model.training
    model.eval()

    images_so_far = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    model.train(mode=was_training)

@torch.no_grad()
def extract_features(fmodel,
                     device,
                     dataloader,
                     pooling='global_avg',
                     classes_name=[],
                     is_plot=True):
    since = time.time()
    if pooling == 'global_avg':
        out = nn.Sequential(fmodel, nn.AdaptiveAvgPool2d((1,1)))

    out = out.to(device)
    out.eval()

    X_data = np.empty((len(dataloader.dataset), 2048))
    Y_data = np.empty((len(dataloader.dataset)))
    print(dataloader.batch_size, X_data.shape, len(dataloader.dataset))
    ind = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = torch.squeeze(out(inputs))
        _b = outputs.shape[0]
        #X_data[ind:ind+_b,] = outputs.cpu().detach().numpy()
        X_data[ind:ind + _b, ] = outputs.cpu().numpy()
        Y_data[ind:ind+_b,] = labels.cpu().numpy()
        ind += _b

        print(outputs.shape, type(outputs), labels.shape, type(labels))

    print(X_data.shape, type(X_data), Y_data.shape, type(Y_data))

    time_elapsed = time.time() - since
    print('Extract features in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("saving feature data of X:{} and Y:{} into npy".format(X_data.shape, Y_data.shape))
    np.save("x_features.npy", X_data)
    np.save("y_label.npy", Y_data)

    if is_plot:
        visualize.features_exam(X_data, Y_data, classes_name=classes_name)

@torch.no_grad()
def feature2classifier_image(fmodel, device,
                             classifier,
                             img_path,
                             transforms=None,
                             height=224, width=224,
                             pooling='global_avg',
                             classes_name=[]):
    """
    :param fmodel:
    :param device:
    :param classifier:
    :param img_path:
    :param transforms:
    :param height:
    :param width:
    :param pooling:
    :param classes_name:
    :return:
        [img_path, predId, pred[predId], classes_name[predId]] ex["./xxx.jpg", 0, 0.99, "lab1"]
    """
    since = time.time()
    if pooling == 'global_avg':
        out = nn.Sequential(fmodel, nn.AdaptiveAvgPool2d((1, 1)))

    out = out.to(device)
    out.eval()

    img = img2torch(img_path, transforms, height, width)
    input_img = img.to(device)
    output = torch.squeeze(out(input_img))
    print(output.shape, type(output))
    output = output.cpu().numpy()

    result = []
    if classifier[2] == 'xgboost':
        _d = classifier[1](np.expand_dims(output, axis=0))
        pred = classifier[0].predict(_d)
        predId = np.argmax(pred, axis=1)
        print(pred, np.argmax(pred, axis=1))
        result = [img_path, predId, pred[predId], classes_name[predId]]

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return result

@torch.no_grad()
def feature2classifier(fmodel,
                       device,
                       classifier,
                       dataloader=None,
                       pooling='global_avg',
                       classes_name=[]):
    """
    :param fmodel:
    :param device:
    :param dataloader:
    :param classifier: (cls_model, input_fun, model_type) ex(xgbmodel, xgb.DMatrix, "xgboost")
    :param pooling:
    :param classes_name:
    :return:
        [[label, predicted, score], ...] ex[[0, 0, 0.99], [0, 1, 0.80]]
    """
    since = time.time()
    if pooling == 'global_avg':
        out = nn.Sequential(fmodel, nn.AdaptiveAvgPool2d((1, 1)))

    out = out.to(device)
    out.eval()

    results = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = torch.squeeze(out(inputs))
        #outputs = outputs.cpu().detach().numpy()
        outputs = outputs.cpu().numpy()
        labels = labels.cpu().numpy()
        #print(outputs.shape, type(outputs))
        #print(labels)

        if classifier[2] == 'xgboost':
            _d = classifier[1](outputs)
            preds = classifier[0].predict(_d)
            predIds = np.argmax(preds, axis=1)
            _results = [[labels[_i], _d, preds[_i][_d]] for _i, _d in enumerate(predIds)]
            #print(preds, np.argmax(preds, axis=1))

        results += _results

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print(results)
    return results


def train_model(model, device,
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
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    torch.save(best_model_wts, "best_model_wts")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model