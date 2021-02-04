import sys
import os
import json
import traceback
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
    #iaa.Grayscale(alpha=(0.0, 1.0)),
    #iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    #iaa.Affine(rotate=(-10, 10)),
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


def mask2poly(pos):
    cnts, hier = cv2.findContours(pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 1:
        points = []
        for c in cnts:
            _cnt = []
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            for p in range(approx.shape[0]):
                _cnt.append([int(approx[p][0][0]), int(approx[p][0][1])])
            points.append(_cnt)
    elif len(cnts) == 1:
        points = []
        peri = cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], 0.01 * peri, True)
        for p in range(approx.shape[0]):
            points.append([int(approx[p][0][0]), int(approx[p][0][1])])
    else:
        points = []

    return points


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


def app_load_image():
    null = None
    datasetRoot = "./data/eval_damages_labeled/seg_test"
    pretrained_path = "./weights/D_20201216/mrcnn_cd_aug_15.pth"
    #"./weights/mrcnn_cd_20200908_101_aug_14.pth" #"./weights/mrcnn_cd_20200821_aug_10.pth" #"./weights/mrcnn_cd_20200820_14.pth"

    #CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR", "CDBL", \
    #         "CFFR", "CFFL", "CFBR", "CFBL", "CS", "CMR", "CML", \
    #         "CLF", "CLB", "CWF", "CWB", "CG", "CTA", "CP"]

    CLASS = ['DS', 'DD', 'DC', 'DW', 'DH', 'DN', 'DR', 'CTA', 'CP'] #['D']  # DN(None), DR(reflect)
    DIM = 1024
    PAD = 32

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    mrcnn = mask_rcnn.MaskRCNN(backbone='resnet50',
                               anchor_ratios=(0.33, 0.5, 1, 2, 3),
                               num_classes=1 + len(CLASS))

    mrcnn.load_state_dict(torch.load(pretrained_path))
    mrcnn.to(device)
    print(mrcnn, datasetRoot)

    for _root, dirs, files in os.walk(datasetRoot):
        #print(_root, dirs, files)
        if os.name == 'nt':
            folder = mrcnnDataGen.ntPath(_root)
        else:
            folder = _root

        for f in files:
            if ".json" in f:
                continue

            labelmeDict = {"version":"4.5.6", "flags":{}, "shapes":[], \
                           "imagePath":"", "imageData":null, "imageHeight":0, "imageWidth":0}            
            
            fpath = mrcnnDataGen.path_join(folder, f)
            print("fpath: {}".format(fpath))
            try:
                pre_result = engine.evaluate_image(mrcnn, fpath, device, CLASS, [True,DIM,DIM], [False,PAD],
                                                   score_threshold=0.7,
                                                   is_plot=True, plot_folder='./plotTest_D')

                print(pre_result.keys(), pre_result['labels'], pre_result['scores'], pre_result['class_names'], len(pre_result['masks']))
                print("masks shape:{}".format(pre_result['masks'].shape))
                # generate labelme json format
                NP_WHERE_MASK = 0.5
                img_org = cv2.imread(fpath)
                #img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                org_shape = img_org.shape
                img_h, img_w = img_org.shape[0:2]
                masks = pre_result['masks']
            
                labelmeDict["imagePath"] = os.path.basename(fpath)
                labelmeDict["imageHeight"] = img_h
                labelmeDict["imageWidth"] = img_w
                for i in range(masks.shape[0]):
                    pos = np.zeros((org_shape[0], org_shape[1]))
                    #print("i:{}, shape:{}".format(i, pos.shape))
                    cols, rows = np.where(masks[i,:,:,0] > NP_WHERE_MASK)

                    pos[cols, rows] = 255
                    pos = pos.astype(np.uint8)
                    points = mask2poly(pos)
                
                    label = pre_result['class_names'][pre_result['labels'][i]]
                    print("label{}:{}, points:{}".format(i, label, points))
                    _nd_points = np.array(points)
                    print(_nd_points.shape, len(_nd_points.shape))
                    if len(_nd_points.shape) in [1, 3]:
                        max_len, max_len_id = 0, 0
                        for _i in range(_nd_points.shape[0]):
                            if len(points[_i]) > max_len:
                                max_len = len(points[_i])
                                max_len_id = _i

                        print(max_len, max_len_id)
                        points = points[max_len_id]


                    polyObj = {"label": "", "points": [], "group_id": null, "shape_type": "polygon", "flags": {}}
                    polyObj["label"] = label
                    for p in points:
                        polyObj["points"].append(p)

                    labelmeDict["shapes"].append(polyObj)
                
                    #points = np.array(points)
                    #points = np.reshape(points, (points.shape[0], 1, points.shape[1]))
                    #cv2.drawContours(img_org, [points.astype(int)], -1, (255), 5)

                #cv2.imwrite("tt.jpg", img_org)
                _path = fpath.split('.jpg')[0]
                json_path = _path + ".json"
                print("save labelme path:{}".format(json_path))
                json_data = json.dumps(labelmeDict, indent=2, separators=(',', ': '))
                with open(json_path, 'w') as fid:
                    fid.write(json_data)

            except Exception as e:
                print("fpath: {} not work!".format(fpath))
                error_class = e.__class__.__name__
                detail = e.args[0]
                cl, exc, tb = sys.exc_info()
                lastCallStack = traceback.extract_tb(tb)[-1]
                fileName = lastCallStack[0]
                lineNum = lastCallStack[1]
                funcName = lastCallStack[2]
                errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
                print(errMsg)

            #sys.exit(1)


def app(apptype='train'):
    datasetRoot = "./data/eval_damages_labeled/seg_test" #"./data/Cathay_Damage_Training/20201117"
    pretrained_path = "./weights/D_20201216/mrcnn_cd_aug_15.pth"
    #"./weights/mrcnn_cd_20200908_101_aug_14.pth" #"./weights/mrcnn_cd_20200820_14.pth"

    #CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR", "CDBL", \
    #         "CFFR", "CFFL", "CFBR", "CFBL", "CS", "CMR", "CML", \
    #         "CLF", "CLB", "CWF", "CWB", "CG", "CTA", "CP"]

    CLASS = ['DS', 'DD', 'DC', 'DW', 'DH', 'DN', 'DR', 'CTA', 'CP']

    # ==== parameters ==== #
    BATCHSIZE = 2
    if apptype == 'eval':
        AUG = None
        SHUFFLE = False
    else:
        AUG = SEQ_AUG
        SHUFFLE = True
    DIM = 1024
    PAD = 32
    EPOCHS = 150
    # =================== #

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # augmentation=SEQ_AUG/None
    dataset = mrcnnDataGen.MRCnnDataset(datasetRoot,
                                        resize_img=[True, DIM, DIM],
                                        padding=[True, PAD],
                                        augmentation=AUG,
                                        pil_process=True,
                                        npy_process=False,
                                        transforms=transforms,
                                        page_classes=CLASS,
                                        field_classes=[],
                                        data_type='page')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=SHUFFLE, collate_fn=obj_utils.collate_fn)

    # resnet50
    mrcnn = mask_rcnn.MaskRCNN(backbone='resnet50',
                               anchor_ratios=(0.33, 0.5, 1, 2, 3),
                               num_classes=1 + len(CLASS))

    mrcnn.load_state_dict(torch.load(pretrained_path))
    mrcnn.to(device)
    print(mrcnn)

    if apptype == 'eval':
        engine.evaluate(mrcnn, dataloader, device, class_names=CLASS, is_plot=True, plot_folder='./plotEval_damages')
        print("evaluation done")
        return


    # construct an optimizer
    params = [p for p in mrcnn.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.94)

    # let's train it for 10 epochs
    save_model_prefix = './weights/D_20201125/mrcnn_cd'+'_aug_'
    print("save model to: {}".format(save_model_prefix))
    for epoch in range(EPOCHS):
        lr_scheduler.step()
        engine.train_one_epoch(mrcnn, optimizer, dataloader, device, epoch, print_freq=10)
        if (epoch+1) % 10 == 0:
            save_model_name = save_model_prefix + str((epoch+1)//10) + '.pth'
            torch.save(mrcnn.state_dict(), save_model_name)

    #extractBBox(dataset, dataloader, CLASS)
    #plotMask(dataset, dataloader, CLASS)


if __name__ == '__main__':
    args = cathay_utils.parse_commands()

    #app(apptype=args.mode)
    app_load_image()
    #app_color()