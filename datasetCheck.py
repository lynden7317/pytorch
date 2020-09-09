import os
import sys
import time
import torch
import torchvision
import pandas as pd

from dataloader import mrcnnDataGen

from obj_detection import utils as obj_utils

def prepare_excel(classes, ):
    root = "./dataset_excel"
    if not os.path.isdir(root):
        os.makedirs(root)
    timestr = time.strftime("%Y-%m-%d", time.localtime())
    fname = os.path.join(root, timestr+'.xlsx')
    writer = pd.ExcelWriter(fname)

    writer_seq = ['ID', 'DATE', '#CARs', '#DAMAGEs'] + classes
    data_empty = {}
    for s in writer_seq:
        data_empty[s] = [' ']

    df_empty = pd.DataFrame(data=data_empty)
    df_empty.to_excel(writer, 'CASES', startrow=0, index=False, columns=writer_seq)

    writer.save()
    row_count = 1
    return writer, writer_seq, row_count

def data2Excel(di, writer, page, row_count, columns=None):
    df_i = pd.DataFrame(data=di)
    if columns is None:
        df_i.to_excel(writer, page, startrow=row_count, header=False, index=False)
    else:
        df_i.to_excel(writer, page, startrow=row_count, header=False, index=False, columns=columns)
    writer.save()


def app():
    datasetRoot = "./data/car_loc/"
    CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR", "CDBL", "CFFR", "CFFL", "CFBR", "CFBL", "CC", "CP", "CL", "D"]
    BATCHSIZE = 4
    DIM = 1024
    PAD = 32

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = mrcnnDataGen.MRCnnDataset(datasetRoot,
                                        resize_img=[False, DIM, DIM],
                                        padding=[False, PAD],
                                        augmentation=None,
                                        pil_process=True,
                                        npy_process=False,
                                        transforms=transforms,
                                        page_classes=CLASS,
                                        field_classes=[],
                                        data_type='page')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False,
                                             collate_fn=obj_utils.collate_fn)


    excel_writer, excel_seq, excel_row = prepare_excel(CLASS)

    dataDict = {}
    classDict = dict(enumerate(CLASS, start=1))
    timestr = time.strftime("%Y-%m-%d", time.localtime())
    for images, targets in dataloader:
        for _i, v in enumerate(targets):
            #print(_i, v.keys(), v['image_id'], v['labels'])
            dataset_id = v['image_id']
            labels = v['labels'].cpu().clone().numpy()
            img_path = dataset.imgs[dataset_id]
            basename = os.path.basename(img_path)
            img_id = int(basename.split('.jpg')[0])
            labelNames = [classDict[l] for l in labels]
            #print(labelNames)
            dataDict[img_id] = {"DATE": timestr, "LABELS":labelNames}

    di = {}
    for s in excel_seq:
        di[s] = [' ']
    for i in sorted(dataDict.keys()):
        dc = dict.fromkeys(CLASS, 0)
        totalCs = 0
        totalDs = 0
        di['ID'] = [str(i)+'.jpg']
        di['DATE'] = [dataDict[i]['DATE']]
        for _l in dataDict[i]['LABELS']:
            if _l[0] == 'C':
                totalCs += 1
            if _l[0] == 'D':
                totalDs += 1
            dc[_l] += 1
        di['#CARs'] = [totalCs]
        di['#DAMAGEs'] = [totalDs]
        for _c in CLASS:
            di[_c] = [dc[_c]]
        print(di)
        data2Excel(di, excel_writer, 'CASES', excel_row, columns=excel_seq)
        excel_row += 1

    excel_writer.close()


if __name__ == '__main__':
    app()