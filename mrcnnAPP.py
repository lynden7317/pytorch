import sys
import torch

from dataloader import mrcnnDataGen
from obj_detection import mask_rcnn
from obj_detection import engine
from obj_detection import utils as obj_utils


if __name__ == '__main__':
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

    dataset = mrcnnDataGen.MRCnnXMLDataset(datasetRoot,
                                           page_classes=page_classes,
                                           field_classes=[],
                                           data_type='page')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=obj_utils.collate_fn)
    print(dataloader.batch_size)
    #print(dataset.imgs)
    #print(dataset.imgData[0])
    #sys.exit(1)

    mrcnn = mask_rcnn.MaskRCNN(num_classes=len(page_classes)+1)
    #print(mrcnn)
    mrcnn.to(device)

    mrcnn.load_state_dict(torch.load(mrcnn_pth))

    """
    coco_evaluator = engine.gen_coco_evaluator(mrcnn, dataloader)
    print(coco_evaluator)
    coco_evaluator, eval_values = engine.evaluate_coco(mrcnn, dataloader, device=device, coco_evaluator=coco_evaluator)
    print("eval_values: {}".format(eval_values))
    """

    #engine.evaluate(mrcnn, dataloader, device, class_names=page_classes, is_plot=True)

    imgpath = 'img_test.jpg'
    engine.evaluate_image(mrcnn, imgpath, device, class_names=page_classes, is_plot=True)