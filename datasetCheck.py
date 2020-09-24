import os
import logging
import sys
import shutil
import time
import torch
import torchvision
import pandas as pd

import cathay_utils

from dataloader import mrcnnDataGen

from obj_detection import utils as obj_utils

# logging.DEBUG
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M",
                    handlers=[logging.FileHandler('case_division.log','w','utf-8')])

def case_division(root):
    num_in_case = 3
    upper_folder = os.path.dirname(root)
    division_folder = os.path.join(upper_folder, "case_division")
    division_folder = cathay_utils.nt_path(division_folder)
    if not os.path.isdir(division_folder):
        try:
            os.makedirs(division_folder)
        except:
            logging.error("cannot create folder at path:{}".format(division_folder))
            return

    case_counter = 0
    for rs, ds, fs in os.walk(root):
        if os.name == 'nt':
            folder = cathay_utils.nt_path(rs)
        else:
            folder = rs
        logging.info("parsing folder: {}".format(folder))
        print("parsing folder: {}".format(folder))
        folder_name = folder.split("/")[-1]
        lab1 = folder_name.split("_")[0]
        lab2 = folder_name.split("_")[-1]
        for f in fs:
            fpath = cathay_utils.path_join(folder, f)
            file_name = os.path.basename(fpath).split(".jpg")[0]
            new_name = lab1+"_"+lab2+"_"+file_name+".jpg"
            sub_case_folder = division_folder + "/" + str(int(case_counter/num_in_case))
            if not os.path.isdir(sub_case_folder):
                try:
                    os.makedirs(sub_case_folder)
                except:
                    logging.error("cannot create folder at path:{}".format(sub_case_folder))
                    sub_case_folder = division_folder

            new_path = os.path.join(sub_case_folder, new_name)
            logging.info("division {} to {}".format(fpath, new_path))
            shutil.copy(fpath, new_path)
            case_counter += 1

def prepare_excel(classes, ):
    root = "./dataset_excel"
    if not os.path.isdir(root):
        os.makedirs(root)
    timestr = time.strftime("%Y-%m-%d", time.localtime())
    fname = os.path.join(root, timestr+'.xlsx')
    writer = pd.ExcelWriter(fname)

    writer_seq = ['CASE', 'ID', 'DATE', 'NAME', '#CARs', '#DAMAGEs'] + classes
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


def app(root):
    CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", \
             "CDFL", "CDBR", "CDBL", "CFFR", "CFFL", \
             "CFBR", "CFBL", "CS", "CMR", "CML", \
             "CCFR", "CCFL", "CCBR", "CCBL", "CLF", \
             "CLB", "CL", "CGA", "CGB", "CP", \
             "DS", "DD", "DC", "DW", "DH"]
    BATCHSIZE = 4
    DIM = 1024
    PAD = 32

    datasetRoot = root
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
            people_name = os.path.dirname(img_path).split('/')[-1]
            basename = os.path.basename(img_path)
            #img_id = int(basename.split('.jpg')[0])
            img_id = basename.split('.jpg')[0]
            labelNames = [classDict[l] for l in labels]
            #print(labelNames)
            dataDict[img_id] = {"DATE": timestr, "LABELS":labelNames, "NAME": people_name}

    di = {}
    for s in excel_seq:
        di[s] = [' ']
    for i in sorted(dataDict.keys()):
        print(i)
        dc = dict.fromkeys(CLASS, 0)
        totalCs = 0
        totalDs = 0
        _name = i.split("_")
        case_name = _name[0]+"_理賠勘車_"+_name[1]
        img_name = _name[2]
        di['CASE'] = [case_name]
        di['ID'] = [img_name+'.jpg']
        di['DATE'] = [dataDict[i]['DATE']]
        di['NAME'] = [dataDict[i]['NAME']]
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
    #root = "./data/case_division/"
    #app(root)

    root = "./data/cathay_real"
    case_division(root)