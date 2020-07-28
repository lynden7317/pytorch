import sys
import numpy as np
import torch
import torchvision

from torch.utils.data import DataLoader

from dataloader import mrcnnDataGen
from dataloader import imgDataGen
from obj_detection import mask_rcnn
from obj_detection import engine
from obj_detection import utils as obj_utils
from obj_detection import visualize

import cv2

import warnings
warnings.filterwarnings("ignore")

def app2():
    datasetRoot = "./data/ctbc"
    pretrained_path = './training/mrcnn_model_large_page_100.pth' #'./training/mrcnn_model_large_page.pth' #'./training/maskrcnn_large_page_model.pth'

    CTBCPAGEID = ['B0060101', 'B0060201', 'B0060301', 'B0060401', 'B0060501', 'b0060101', 'b0060901', 'B7000101', 'B7000102', 'B7000103', 'B7000104', 'B7000201', 'B7000202', 'B7000203', \
                  'B7000301', 'b7000101', 'b7000102', 'b7000901', 'B8080201', 'B8080301', 'B8080302', 'B8080501', 'B8080502', 'b8080101', 'b8080901', 'B0040101', 'B0040102', 'B0040103', \
                  'B0040201', 'B0040301', 'B0040302', 'B0040305', 'B0040401', 'B0040402', 'B0040403', 'b0040301', 'b0040302', 'B0130101', 'B0130102', 'B0130201', 'B0130301', 'B0130302', \
                  'B0130303', 'B0130304', 'B0130305', 'B0130306', 'B0130601', 'B0130701', 'b0130101', 'B0120101', 'B0120102', 'B0120103', 'b0120101', 'b0120102', 'b0120103', 'b0120104', \
                  'B0080101', 'B0080301', 'B0080302', 'B0080303', 'B0080304', 'B0080306', 'B0080307', 'B0080501', 'B0080601', 'B0080701', 'b0080101', 'b0080501', 'b0080901', 'B0050101', \
                  'B0050102', 'B0050301', 'B0050302', 'B0050303', 'B0050601', 'b0050101', 'b0050901', 'B8120101', 'B8120201', 'B8120301', 'B8120601', 'B8120701', 'b8120101', 'b8120201', \
                  'B0070101', 'B0070102', 'B0070301', 'B0070302', 'B0070601', 'b0070101', 'b0070301', 'B0090101', 'B0090102', 'B0090103', 'B0090301', 'B0090302', 'B0090303', 'B0090601', \
                  'B0090701', 'b0090101', 'b0090301', 'B0170101', 'B0170102', 'B0170201', 'B0170301', 'B0170302', 'B0170601', 'B0170602', 'B0170701', 'b0170101', 'b0170201', 'B0210301', \
                  'B0210303', 'B0210304', 'B0210305', 'B0210306', 'B0210307', 'B0210308', 'B0210701', 'b0210101', 'b0210301', 'B1030101', 'B1030102', 'B1030401', 'B1030601', 'b1030101', \
                  'b1030301', 'B8030101', 'B8030301', 'B8030302', 'b8030101', 'B8060101', 'B8060102', 'B8060201', 'B8060202', 'B8060701', 'b8060101', 'b8060901', 'B1080101', 'B1080301', \
                  'b1080101', 'B0500101', 'B0500301', 'B0500601', 'b0500101', 'b0500102', 'b0500901', 'B8070201', 'B8070202', 'B8070301', 'B8070302', 'B8070303', 'b8070101', 'b8070102', \
                  'B8050101', 'B8050301', 'b8050101', 'b8050901', 'B0530101', 'B0530201', 'B0530301', 'B0530302', 'b0530101', 'b0530301', 'B0110101', 'B0110301', 'B0110302', 'B0110303', \
                  'B0110401', 'b0110101', 'B0520101', 'B0520201', 'B0520301', 'b0520301', 'b0520901', 'K0060101', 'K0060201', 'K0060301', 'k0060101', 'k0060901', 'K7000101', 'K7000103', \
                  'K7000104', 'K7000201', 'K7000202', 'k7000101', 'k7000102', 'k7000901', 'K8080201', 'K8080301', 'K8080501', 'k8080101', 'k8080901', 'K0040101', 'K0040102', 'K0040103', \
                  'K0040301', 'K0040302', 'K0040305', 'K0040401', 'K0040403', 'k0040301', 'k0040302', 'K0130101', 'K0130102', 'K0130301', 'K0130302', 'K0130303', 'K0130304', 'K0130306', \
                  'K0130701', 'k0130101', 'K0120101', 'k0120101', 'K0080101', 'K0080301', 'K0080303', 'K0080306', 'K0080501', 'K0080701', 'k0080101', 'K0050101', 'K0050301', 'K0050302', \
                  'k0050101', 'K8120101', 'K8120201', 'K8120301', 'K8120601', 'k8120101', 'k8120201', 'K0070101', 'K0070301', 'K0070601', 'k0070101', 'k0070301', 'K0090101', 'K0090102', \
                  'K0090103', 'K0090301', 'K0090302', 'K0090303', 'k0090101', 'k0090301', 'K0170101', 'K0170301', 'K0170302', 'k0170101', 'k0170201', 'K0210307', 'k0210101', 'k0210301', \
                  'K1030101', 'K1030102', 'K1030402', 'k1030101', 'K8030101', 'K8030301', 'k8030101', 'K8060101', 'K8060201', 'K8060202', 'K8060701', 'k8060101', 'K1080101', 'k1080101', \
                  'K0500101', 'K0500301', 'k0500101', 'k0500102', 'K8070301', 'K8070302', 'K8070303', 'k8070101', 'k8070102', 'K8050101', 'K8050301', 'k8050101', 'K0530101', 'K0530301', \
                  'k0530101', 'k0530301', 'K0110101', 'K0110301', 'K0110302', 'K0110303', 'k0110101', 'K0520101', 'K0520301', 'k0520301', 'T0000001', 'S0000001', 'I00100A', 'I00200A', \
                  'I00300A', 'I00400A', 'I00500A', 'I00600A']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    import imgaug.augmenters as iaa
    seq_aug = iaa.Sequential([
        iaa.Crop(px=(0, 16)),
        iaa.Fliplr(1.0)
    ])

    #print(1+len(CTBCPAGEID))
    p_model = mask_rcnn.MaskRCNN(backbone='resnet50',
                                 anchor_ratios=(0.333333, 0.5, 1, 2, 3),
                                 num_classes=1+len(CTBCPAGEID))

    #p_model = mask_rcnn.MaskRCNN(num_classes=1+len(CTBCPAGEID))

    p_model.load_state_dict(torch.load(pretrained_path))
    #print(p_model.backbone)
    #new_model = p_model.backbone.body
    #print(new_model)
    p_model.to(device)
    print(p_model)

    #import img_process
    #from tools import img_process
    from dataloader import img_utils
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    #transforms = torchvision.transforms.Compose([
    #                     torchvision.transforms.ToTensor(),
    #                     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = mrcnnDataGen.MRCnnXMLDataset(datasetRoot,
                                           resize_img=[False, 512, 512],
                                           padding=[True, 32],
                                           augmentation=seq_aug,
                                           pil_process=True,
                                           npy_process=False,
                                           transforms=transforms,
                                           page_classes=CTBCPAGEID,
                                           field_classes=[],
                                           data_type='page')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=obj_utils.collate_fn)

    imgpath = 'bank_test.jpg'
    engine.evaluate_image(p_model, imgpath, device,
                          resize_img=[True, 512, 512],
                          padding=[True, 32],
                          class_names=CTBCPAGEID, is_plot=True)

    """
    from dataloader import img_utils
    p_model.eval()
    for inputs, targets in dataloader:
        # Make a grid from batch
        print(type(inputs), len(inputs), inputs[0].shape)
        print(type(targets), targets[0].keys(), targets[0]['masks'].shape)

        img = torchvision.transforms.ToPILImage()(inputs[1])
        img_org = np.array(img)
        original_img_shape = img_org.shape
        print("ori shape: {}".format(original_img_shape))

        padding = [(32, 32), (32, 32), (0, 0)]
        img_pad = np.pad(img_org, padding, mode='constant', constant_values=0)
        img_pad_shape = img_pad.shape
        print("img_pad: {}".format(img_pad.shape))
        img_resize, window, scale, resize_padding, crop = img_utils.resize_image(img_pad, min_dim=512, max_dim=512)
        img_resize_shape = img_resize.shape
        print("scale: {}, resize_padding: {}".format(scale, resize_padding))

        _img_resize = torchvision.transforms.ToTensor()(img_resize)
        _img_resize = torch.unsqueeze(_img_resize, 0)
        _img_resize = _img_resize.to(device)
        output = p_model(_img_resize)[0]
        print(output.keys(), output['masks'].shape, output['scores'])

        boxes = output['boxes'].detach().cpu().clone().numpy()
        labels = output['labels'].detach().cpu().clone().numpy()
        scores = output['scores'].detach().cpu().clone().numpy()
        masks = output['masks'].detach().cpu().clone().numpy()
        masks = masks.transpose((0, 2, 3, 1))
        print(boxes[0], masks[0].shape)


        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        print(boxes, "window", window)
        wx1, wy1, wx2, wy2 = window
        shift = np.array([wx1, wy1, wx1, wy1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        _scale = np.array([ww, wh, ww, wh])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, _scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = img_utils.denorm_boxes(boxes, img_pad_shape[:2])
        print(boxes, img_pad.shape)

        full_mask = masks[0][:, :, 0]
        full_mask = img_utils.denorm_mask(masks[0][:, :, 0], scale, resize_padding, img_pad_shape[:2])
        #full_mask = img_utils.unmold_mask(masks[0][:, :, 0], boxes[0], img_pad_shape[:2])
        #full_mask = np.where(full_mask >= 0.5, 1, 0).astype(np.bool)
        full_mask = np.expand_dims(full_mask, axis=2)

        color = visualize.random_colors(1)
        #mask_test = np.zeros((img_pad_shape[0], img_pad_shape[1], 1))
        #mask_test[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2]), 0] = 1

        masked_image = img_pad.astype(np.uint32).copy()
        masked_image = visualize.apply_mask(masked_image, full_mask[:, :, 0], color[0])

        plt.imshow(img_pad)
        plt.show()
        plt.imshow(masked_image)
        plt.show()
    """

    """
    # construct an optimizer
    params = [p for p in p_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 100
    for epoch in range(num_epochs):
        engine.train_one_epoch(p_model, optimizer, dataloader, device, epoch, print_freq=10)
        #lr_scheduler.step()

    torch.save(p_model.state_dict(), "mrcnn_model_large_page_100.pth")
    """

    #engine.evaluate(p_model, dataloader, device, class_names=CTBCPAGEID, is_plot=True)


    is_npy = False
    is_pil = True
    for inputs, targets in dataloader:
        # Make a grid from batch
        print(type(inputs), len(inputs), inputs[0].shape)
        print(type(targets), targets[0].keys(), targets[0]['masks'].shape)

        if is_npy:
            img = inputs[0].byte()
            img = img.numpy().transpose((1, 2, 0))
            masked_image = img.astype(np.uint32).copy()

        if is_pil:
            img = torchvision.transforms.ToPILImage()(inputs[0])
            img = np.array(img)
            masked_image = img.astype(np.uint32).copy()

        mask = targets[0]['masks'].byte()
        mask = mask.numpy().transpose((1, 2, 0))
        color = visualize.random_colors(1)
        for _m in range(mask.shape[2]):
            masked_image = visualize.apply_mask(masked_image, mask[:, :, _m], color[0])

        plt.imshow(img)
        plt.show()
        plt.imshow(masked_image)
        plt.show()



    """
    img_path = 'test_bank.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_process.resize_image(img, 512)

    #img = torchvision.transforms.ToPILImage()(img)
    #img = torchvision.transforms.ToTensor()(img)

    img = img_process.moldImage_resnet(img)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()

    img = torch.unsqueeze(img, 0)
    print(img.shape)
    img = img.to(device)

    p_model.eval()
    output = p_model(img)[0]
    print(output)
    #print(type(output), output.keys())
    #print(output['0'].shape, output['1'].shape, output['2'].shape, output['3'].shape)
    """

def app1():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    datasetRoot = "../detection/dataset/test2"
    mrcnn_pth = './training/mrcnn_model_resnet50_air_page.pth'
    page_classes = ['AA001A', 'AA001B', 'AA002A', 'AA003B', 'AA004A', 'AA005B', 'ANA001A', 'ANA001B', 'ANA002A', 'ANA003A', \
                    'CA001A', 'CA001B', 'CD001A', 'CD001B', 'CD002B', 'CD003A', 'CI001A', 'CI001B', 'CP001A', 'CP001B', \
                    'CP002B', 'EM001A', 'EM001B', 'EVA001A', 'EVA001B', 'EVA002A', 'EVA002B', 'JAL001A', 'JAL001B', 'JAL002A', \
                    'JE001A', 'JE001B', 'JE002B', 'JE003B', 'JS001A', 'JS002A', 'JS003A', 'MD001B', 'PA001A', 'PA002A', \
                    'PA002B', 'SC001A', 'SC001B', 'TH001A', 'TH001B', 'TH002A', 'TH002B', 'TIG001B', 'TIG002A', 'TIG002B', \
                    'XI001A', 'XI001B']
    field_classes = ['Name', 'Date', 'Flight', 'FromCity', 'ToCity', 'FromId', 'ToId']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    transforms = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = mrcnnDataGen.MRCnnXMLDataset(datasetRoot,
                                           resize_img=[True, 512, 512],
                                           pil_process=True,
                                           npy_process=False,
                                           transforms=transforms,
                                           page_classes=page_classes,
                                           field_classes=field_classes,
                                           data_type='page_field')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=obj_utils.collate_fn)
    print(dataloader.batch_size)
    # print(dataset.imgs)
    # print(dataset.imgData[0])
    # sys.exit(1)

    #mrcnn = mask_rcnn.MaskRCNN(num_classes=91) # len(page_classes) + 1
    # print(mrcnn)
    #mrcnn.to(device)

    #mrcnn.load_state_dict(torch.load(mrcnn_pth))

    """
    coco_evaluator = engine.gen_coco_evaluator(mrcnn, dataloader)
    print(coco_evaluator)
    coco_evaluator, eval_values = engine.evaluate_coco(mrcnn, dataloader, device=device, coco_evaluator=coco_evaluator)
    print("eval_values: {}".format(eval_values))
    """

    # engine.evaluate(mrcnn, dataloader, device, class_names=page_classes, is_plot=True)

    #imgpath = 'img_test.jpg'
    #engine.evaluate_image(mrcnn, imgpath, device, class_names=page_classes, is_plot=True)

    is_npy = False
    is_pil = True
    for inputs, targets in dataloader:
        # Make a grid from batch
        print(type(inputs), len(inputs), inputs[0].shape)
        print(type(targets), targets[0].keys(), targets[0]['masks'].shape)

        if is_npy:
            img = inputs[0].byte()
            img = img.numpy().transpose((1,2,0))
            masked_image = img.astype(np.uint32).copy()

        if is_pil:
            img = torchvision.transforms.ToPILImage()(inputs[0])
            img = np.array(img)
            masked_image = img.astype(np.uint32).copy()

        mask = targets[0]['masks'].byte()
        mask = mask.numpy().transpose((1, 2, 0))
        color = visualize.random_colors(1)
        for _m in range(mask.shape[2]):
            masked_image = visualize.apply_mask(masked_image, mask[:,:,_m], color[0])

        plt.imshow(img)
        plt.show()
        plt.imshow(masked_image)
        plt.show()


if __name__ == '__main__':
    app2()