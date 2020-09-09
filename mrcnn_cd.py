import sys
import os
import random
import numpy as np
import torch
import torchvision
import cv2

from dataloader import mrcnnDataGen
from dataloader import img_utils

from obj_detection import mask_rcnn
from obj_detection import engine
from obj_detection import utils as obj_utils

from obj_detection import visualize
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import cathay_utils


import imgaug.augmenters as iaa
"""
SEQ_AUG = iaa.SomeOf((1, 3), [
    iaa.Grayscale(alpha=(0.0, 1.0)),
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    iaa.Affine(rotate=(-15, 15)),
    iaa.Fliplr(0.5),
    iaa.AddToHue((-50, 50)),
    iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
    iaa.MultiplyHueAndSaturation((0.5, 1.5)),
])
"""
# iaa.Sequential([])
SEQ_AUG = iaa.SomeOf((1, 3), [
    iaa.Grayscale(alpha=(0.0, 1.0)),
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    iaa.Affine(rotate=(-10, 10)),
    iaa.AddToHue((-50, 50)),
    iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
    iaa.WithColorspace(
        to_colorspace="HSV",
        from_colorspace="RGB",
        children=iaa.WithChannels(
            0, iaa.Add((0, 50)))),
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
    iaa.MultiplyHueAndSaturation((0.5, 1.5)),
])

def extractColors(mask):
    pass


def plotMask(dataset, dataloader, class_names):
    is_pil = True
    for inputs, targets in dataloader:
        for batch in range(len(inputs)):
            if is_pil:
                img = torchvision.transforms.ToPILImage()(inputs[batch])
                img = np.array(img)
                masked_image = img.astype(np.uint32).copy()

            boxes = targets[batch]["boxes"].cpu().clone().numpy()
            labels = targets[batch]["labels"].cpu().clone().numpy()-1
            masks = targets[batch]["masks"].cpu().clone().numpy()
            masks = np.expand_dims(masks, -1)
            print(boxes.shape, labels.shape, masks.shape)

            img_id = targets[batch]['image_id'].byte().numpy()[0]
            img_path = dataset.imgs[img_id]
            img_name = os.path.basename(img_path)
            name = "./damages/" + img_name
            visualize.display_instances(img, boxes,
                                        masks, labels,
                                        class_names,
                                        is_display=False,
                                        is_save=[True, name])

            """
            mask = targets[batch]['masks'].byte()
            mask = mask.numpy().transpose((1, 2, 0))
            img_id = targets[batch]['image_id'].byte().numpy()[0]
            img_path = dataset.imgs[img_id]
            img_name = os.path.basename(img_path)
            color = visualize.random_colors(mask.shape[2])

            for _m in range(mask.shape[2]):
                masked_image = visualize.apply_mask(masked_image, mask[:, :, _m], color[_m])

            name = "./damages/" + img_name
            print("save: {}".format(name))
            plt.imshow(masked_image)
            plt.savefig(name)
            """



def extractBBox(dataset, dataloader, class_name):
    is_pil = True
    report = {}
    class_name = ['BG']+class_name
    for inputs, targets in dataloader:
        # Make a grid from batch
        print("input: ", type(inputs), len(inputs))
        #print("target: ", type(targets), targets[batchid].keys(), targets[batchid]['masks'].shape)
        for _id in range(len(inputs)):
            if is_pil:
                img = torchvision.transforms.ToPILImage()(inputs[_id])
                img = np.array(img)

            mask = targets[_id]['masks'].byte()
            mask = mask.numpy().transpose((1, 2, 0))
            img_id = targets[_id]['image_id'].byte().numpy()[0]
            img_path = dataset.imgs[img_id]
            img_name = os.path.basename(img_path)
            prefix = img_name.split('.jpg')[0]
            report[int(prefix)] = {"name": img_name, "damages":[]}
            current = report[int(prefix)]
            print(img.shape, mask.shape, img_id, dataset.imgs[img_id])
            for _m in range(mask.shape[2]):
                #print(targets[_id]['labels'][_m].byte())
                lab = class_name[targets[_id]['labels'][_m].byte()]
                print("lab: {}".format(lab))
                pos = np.where(mask[:,:,_m]>0)
                #print(pos[0].shape, pos[1].shape)
                #sys.exit(1)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                # append(label, mask_area, bbox_area, bbox_shape)
                current["damages"].append((lab, pos[0].shape[0],
                                           (xmax-xmin)*(ymax-ymin), str(xmax-xmin)+"x"+str(ymax-ymin)))
                masked_img = img[ymin:ymax, xmin:xmax]
                print(masked_img.shape)
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)

                #if pos[0].shape[0] > 20000:
                name = "./plotDamage/"+prefix+"_"+lab+"_"+str(len(current["damages"]))+".jpg"
                print("save: {}".format(name))
                cv2.imwrite(name, masked_img)

    # save the report
    saveName = "./plotDamage/report.txt"
    with open(saveName, 'w') as fid:
        for _k in sorted(report.keys()):
            iname = report[_k]["name"]
            dnum = len(report[_k]["damages"])
            write_str = str(_k)+" "+iname+" "+str(dnum)+" "
            for _d in report[_k]["damages"]:
                _str = write_str + str(_d[0]) + " " + str(_d[1]) + " " + str(_d[2]) + " " + str(_d[3]) + "\n"
                fid.write(_str)


def app_color():
    datasetRoot = "./data/car_loc/"  # "./data/damage_20200826/" #"./data/damage_0827/"
    pretrained_path = "./weights/mrcnn_cd_20200908_101_aug_14.pth"  # "./weights/mrcnn_cd_20200820_14.pth"
    CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR", "CDBL", "CFFR", "CFFL", "CFBR", "CFBL", "CC", "CP", "CL"]
    # CLASS = ['DS', 'DD', 'DC', 'DW', 'DH'] #['D']
    BATCHSIZE = 4
    DIM = 1024
    PAD = 32
    EPOCHS = 150

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = mrcnnDataGen.MRCnnDataset(datasetRoot,
                                        resize_img=[True, DIM, DIM],
                                        padding=[True, PAD],
                                        augmentation=None,
                                        pil_process=True,
                                        npy_process=False,
                                        transforms=transforms,
                                        page_classes=CLASS,
                                        field_classes=[],
                                        data_type='page')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=obj_utils.collate_fn)

    is_pil = True
    outputfolder = "./colors/"
    for inputs, targets in dataloader:
        for _id in range(len(inputs)):
            img_id = targets[_id]['image_id'].byte().numpy()[0]
            img_path = dataset.imgs[img_id]
            img_name = os.path.basename(img_path).split(".jpg")[0]
            print(img_path)
            if is_pil:
                img = torchvision.transforms.ToPILImage()(inputs[_id])
                img = np.array(img)

            print(img.shape)
            mask = targets[_id]['masks'].byte()
            mask = mask.numpy().transpose((1, 2, 0))
            for _m in range(mask.shape[2]):
                pos = np.where(mask[:, :, _m] > 0.5)
                print(pos[0].shape, pos[1].shape)

                mask_img = np.zeros(img.shape)
                cv_img = np.zeros(img.shape)

                sample = random.sample(range(pos[0].shape[0]), 300)
                colors = {}
                for p in sample:
                    c = (img[pos[0][p], pos[1][p], 0], img[pos[0][p], pos[1][p], 0], img[pos[0][p], pos[1][p], 0])
                    cname = cathay_utils.closest_colour(c)
                    if cname in colors.keys():
                        colors[cname] += 1
                    else:
                        colors[cname] = 1

                max_color = sorted(colors, key=lambda k: colors[k])
                colorName = max_color[-1]+"_"+max_color[-2]
                print(colorName)

                for c in range(3):
                    mask_img[:, :, c] = np.where(mask[:,:,_m] > 0.5, img[:, :, c], 0)
                #cRGB = (np.sum(np.sum(mask_img, axis=0), axis=0)/pos[0].shape).astype(int)
                #colorName = cathay_utils.closest_colour(cRGB)

                cv_img[:,:,0] = mask_img[:,:,2]
                cv_img[:,:,1] = mask_img[:,:,1]
                cv_img[:,:,2] = mask_img[:,:,0]
                cv2.putText(cv_img, colorName, (10,50), cv2.FONT_HERSHEY_DUPLEX,
                            1, (0, 255, 255), 1, cv2.LINE_AA)
                #print(cRGB, colorName)
                #print(np.sum(np.sum(mask_img, axis=0), axis=0)/pos[0].shape)
                saveName = outputfolder + img_name + "_" + str(_m) + ".jpg"
                cv2.imwrite(saveName, cv_img)




def app_load_image():
    datasetRoot = "./data/val_1/"
    pretrained_path = "./weights/mrcnn_cd_20200908_101_aug_14.pth" #"./weights/mrcnn_cd_20200821_aug_10.pth" #"./weights/mrcnn_cd_20200820_14.pth"
    CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR", "CDBL", "CFFR", "CFFL", "CFBR", "CFBL", "CC", "CP", "CL"]
    #CLASS = ['D'] #['DS', 'DD', 'DC', 'DW', 'DH'] #['D']
    DIM = 1024
    PAD = 32

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    mrcnn = mask_rcnn.MaskRCNN(backbone='resnet101',
                               anchor_ratios=(0.33, 0.5, 1, 2, 3),
                               num_classes=1 + len(CLASS))

    mrcnn.load_state_dict(torch.load(pretrained_path))
    mrcnn.to(device)
    print(mrcnn)

    for _root, dirs, files in os.walk(datasetRoot):
        if os.name == 'nt':
            folder = mrcnnDataGen.ntPath(datasetRoot)
        else:
            folder = _root
        for f in files:
            fpath = mrcnnDataGen.path_join(folder, f)
            print(fpath)
            engine.evaluate_image(mrcnn, fpath, device, CLASS, [True,DIM,DIM], [True,PAD],
                                  score_threshold=0.7,
                                  is_plot=True, plot_folder='./plotTest')



def app():
    datasetRoot = "./data/car_loc/" #"./data/damage_20200826/" #"./data/damage_0827/"
    pretrained_path = "./weights/mrcnn_cd_20200908_101_aug_14.pth" #"./weights/mrcnn_cd_20200820_14.pth"
    CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR", "CDBL", "CFFR", "CFFL", "CFBR", "CFBL", "CC", "CP", "CL"]
    #CLASS = ['DS', 'DD', 'DC', 'DW', 'DH'] #['D']
    BATCHSIZE = 4
    DIM = 1024
    PAD = 32
    EPOCHS = 150

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = mrcnnDataGen.MRCnnDataset(datasetRoot,
                                        resize_img=[True, DIM, DIM],
                                        padding=[True, PAD],
                                        augmentation=None,
                                        pil_process=True,
                                        npy_process=False,
                                        transforms=transforms,
                                        page_classes=CLASS,
                                        field_classes=[],
                                        data_type='page')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=obj_utils.collate_fn)

    # resnet50
    mrcnn = mask_rcnn.MaskRCNN(backbone='resnet101',
                               anchor_ratios=(0.33, 0.5, 1, 2, 3),
                               num_classes=1 + len(CLASS))

    mrcnn.load_state_dict(torch.load(pretrained_path))
    mrcnn.to(device)
    print(mrcnn)

    engine.evaluate(mrcnn, dataloader, device, class_names=CLASS, is_plot=True)

    """
    # construct an optimizer
    params = [p for p in mrcnn.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

    # let's train it for 10 epochs
    save_model_prefix = './weights/mrcnn_cd_20200908_101'+'_aug_'
    for epoch in range(EPOCHS):
        lr_scheduler.step()
        engine.train_one_epoch(mrcnn, optimizer, dataloader, device, epoch, print_freq=10)
        if (epoch+1) % 10 == 0:
            save_model_name = save_model_prefix + str(epoch//10) + '.pth'
            torch.save(mrcnn.state_dict(), save_model_name)
    """
    #extractBBox(dataset, dataloader, CLASS)
    #plotMask(dataset, dataloader, CLASS)

    """
    is_pil = True
    count = 1
    for inputs, targets in dataloader:
        # Make a grid from batch
        batchid = 0
        print("input: ", type(inputs), len(inputs), inputs[batchid].shape)
        print("target: ", type(targets), targets[batchid].keys(), targets[batchid]['masks'].shape)
        sys.exit(1)

        if is_pil:
            img = torchvision.transforms.ToPILImage()(inputs[batchid])
            img = np.array(img)
            masked_image = img.astype(np.uint32).copy()

        mask = targets[batchid]['masks'].byte()
        mask = mask.numpy().transpose((1, 2, 0))
        color = visualize.random_colors(mask.shape[2])
        for _m in range(mask.shape[2]):
            masked_image = visualize.apply_mask(masked_image, mask[:, :, _m], color[_m])

        name = "./plotTest/m_"+str(count)+".jpg"
        print("save: {}".format(name))
        img = masked_image.astype(np.uint8)
        plt.imshow(masked_image)
        plt.savefig(name)
        count += 1
        if count > 20:
            break
    """


if __name__ == '__main__':
    #app()
    #app_load_image()
    app_color()