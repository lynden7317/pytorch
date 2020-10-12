import os
import logging
import sys
import argparse
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

COLS = ['CASE', 'ID', 'PASS', 'TRAIN', 'DATE', 'LABELER', 'CHECKER', '#CARs', '#DAMAGEs']
CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", \
         "CDFL", "CDBR", "CDBL", "CFFR", "CFFL", \
         "CFBR", "CFBL", "CS", "CMR", "CML", \
         "CLF", "CLB", "CL", "CG", "CTA", "CTB", "CP", \
         "DS", "DD", "DC", "DW", "DH"]

def parse_commands():
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # default log folder
    MODEL_DIR = os.path.join(ROOT_DIR, "log")
    # default case folder
    CASE_DIR = os.path.join(ROOT_DIR, "case")


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cathay Car Damage Dataset.')

    parser.add_argument("--txt_table_path", required=False,
                        default="D:/Cathay_DB/txts/cathay_txt_table.txt",
                        metavar="/path/to/txt_table/",
                        help="txt dataset")

    parser.add_argument('--log_path', required=False,
                        default=MODEL_DIR,
                        metavar="/path/to/logs/",
                        help='log path')

    parser.add_argument('--folder', required=False,
                        default=CASE_DIR,
                        metavar="/path/to/target_folder/",
                        help='target folder path')

    parser.add_argument('--labeler', required=False,
                        default="none",
                        metavar="labeler",
                        help='labeler')

    parser.add_argument('--checker', required=False,
                        default="none",
                        metavar="checker",
                        help='checker')

    parser.add_argument('--tag', required=False,
                        default="",
                        metavar="tag",
                        help='tag')

    args = parser.parse_args()

    return args

def get_labs(rs):
    if os.name == 'nt':
        folder = cathay_utils.nt_path(rs)
    else:
        folder = rs
    logging.info("parsing folder: {}".format(folder))
    #print("parsing folder: {}".format(folder))
    folder_name = folder.split("/")[-1]
    lab1 = folder_name.split("_")[0]
    lab2 = folder_name.split("_")[-1]

    return lab1, lab2, folder

def create_txt_table(root):
    upper_folder = os.path.dirname(root)
    out_folder = os.path.join(upper_folder, "txts")
    out_folder = cathay_utils.nt_path(out_folder)
    if not os.path.isdir(out_folder):
        try:
            os.makedirs(out_folder)
        except:
            logging.error("cannot create folder at path:{}".format(out_folder))
            return

    dataDict = {}
    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        for i, f in enumerate(fs):
            fpath = cathay_utils.path_join(folder, f)
            file_name = os.path.basename(fpath).split(".jpg")[0]
            new_name = lab1 + "_" + lab2 + "_" + file_name
            dataDict[new_name] = {'CASE':lab1+"_理賠勘車_"+lab2, 'ID':file_name, 'PASS':'N', 'TRAIN':'N'}


    timestr = time.strftime("%Y-%m-%d", time.localtime())
    txt_name = os.path.join(out_folder, timestr + '.txt')
    with open(txt_name, 'w') as fid:
        for di in dataDict.keys():
            str = ""
            for s in COLS+CLASS:
                str = str + dataDict[di][s] + "\t"
            fid.write(str[:-1]+"\n")

def load_txt_table(txt_path):
    # read txt table
    txt_path = txt_path
    txtDict = {}
    caseDict = {}
    with open(txt_path) as fid:
        for li in fid.readlines():
            li = li.split("\n")[0]
            ls = li.split("\t")
            case = ls[0]
            img_id = ls[1]
            # print(case, img_id)
            _case = case.split("_")[0] + "_" + case.split("_")[2]
            id = case.split("_")[0] + "_" + case.split("_")[2] + "_" + img_id

            txtDict[id] = {}
            for c in COLS+CLASS:
                if c in ['#CARs', '#DAMAGEs']+CLASS:
                    txtDict[id][c] = 0
                else:
                    txtDict[id][c] = "none"
            txtDict[id]["CASE"] = case
            txtDict[id]["ID"] = img_id
            for i, v in enumerate(COLS+CLASS):
                if i < 2:
                    continue
                if i > len(ls)-1:
                    break
                if v in ['#CARs', '#DAMAGEs']+CLASS:
                    txtDict[id][v] = int(ls[i])
                else:
                    txtDict[id][v] = ls[i]

            #txtDict[id]["PASS"] = ls[2]
            #txtDict[id]["TRAIN"] = ls[3]

            if _case in caseDict.keys():
                caseDict[_case].append(img_id)
            else:
                caseDict[_case] = [img_id]

    logging.info("Loading Table Done")
    return txtDict, caseDict

def write2txt(txt_path, txtDict):
    with open(txt_path, 'w') as fid:
        for di in txtDict.keys():
            _str = ""
            for s in COLS+CLASS:
                _str = _str + str(txtDict[di][s]) + "\t"
            fid.write(_str[:-1] + "\n")

def update_txt_table(args, root):
    # read txt table
    updated_path = "D:/Cathay_DB/txts/cathay_txt_table_updated.txt"
    txtDict, caseDict = load_txt_table(args.txt_table_path)

    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        for i, f in enumerate(fs):
            fpath = cathay_utils.path_join(folder, f)
            if ".txt" in fpath:
                note = os.path.basename(fpath).split(".txt")[0]
                case_name = lab1 + "_" + lab2
                for _i in caseDict[case_name]:
                    new_name = lab1 + "_" + lab2 + "_" + _i
                    txtDict[new_name]["PASS"] = "N_("+note+")"
                continue

            file_name = os.path.basename(fpath).split(".jpg")[0]
            new_name = lab1 + "_" + lab2 + "_" + file_name
            txtDict[new_name]["PASS"] = "Y"

    write2txt(updated_path, txtDict)

def load_labeled_data(args):
    updated_path = "D:/Cathay_DB/txts/cathay_txt_table_"+args.tag+".txt"

    txtDict, caseDict = load_txt_table(args.txt_table_path)

    dataloader, dataset = create_dataloader(args.folder)

    dataDict = {}
    classDict = dict(enumerate(CLASS, start=1))
    timestr = time.strftime("%Y-%m-%d", time.localtime())
    for images, targets in dataloader:
        for _i, v in enumerate(targets):
            # print(_i, v.keys(), v['image_id'], v['labels'])
            dataset_id = v['image_id']
            labels = v['labels'].cpu().clone().numpy()
            img_path = dataset.imgs[dataset_id]
            basename = os.path.basename(img_path)
            # img_id = int(basename.split('.jpg')[0])
            img_id = basename.split('.jpg')[0]
            labelNames = [classDict[l] for l in labels]
            # print(labelNames)
            dataDict[img_id] = {"DATE": timestr, "LABELS": labelNames, "LABELER": args.labeler, "CHECKER": args.checker}

    print(len(dataDict), dataDict)
    for id in dataDict.keys():
        for _d in ["DATE", "LABELER", "CHECKER"]:
            txtDict[id][_d] = dataDict[id][_d]
        cars, damages = 0, 0
        for lab in dataDict[id]["LABELS"]:
            if lab[0] == "C":
                cars += 1
            if lab[0] == "D":
                damages += 1
            txtDict[id][lab] += 1

        txtDict[id]["#CARs"] = cars
        txtDict[id]["#DAMAGEs"] = damages

    write2txt(updated_path, txtDict)

def cathay_rename_folder(root):
    dst_folder = "D:\Cathay_DB\Cathay_Image_Database_Refined"

    for rs, ds, fs in os.walk(root):
        lab1, lab2, src_dir = get_labs(rs)
        if src_dir == root:
            continue
        dst_dir = os.path.join(dst_folder, lab1+"_"+lab2)
        print(src_dir, dst_dir)
        shutil.move(src_dir, dst_dir)

def cathay_labeled_parser(root):
    caseDict = {}
    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        if folder == root:
            continue

        case_name = lab1+"_"+lab2
        print("case: {}".format(case_name))
        if case_name not in caseDict.keys():
            caseDict[case_name] = {"imgs":[], "del":[False]}

        for i, f in enumerate(fs):
            fpath = cathay_utils.path_join(folder, f)
            #print(fpath)
            if "txt" in fpath:
                continue

            txt_name = fpath.split(".jpg")[0]+".txt"
            #print(txt_name)
            if os.path.isfile(txt_name):
                try:
                    with open(txt_name, 'r') as fid:
                        li = fid.readline()
                        #print(li)
                        rads = li.split(" ")
                        if int(rads[2]) in [1, 2, 3]:
                            caseDict[case_name]["del"] = [True, int(rads[2])]
                        if int(rads[1]) == 1:
                            caseDict[case_name]["imgs"].append(fpath)

                except:
                    print("cannot open file:{}".format(txt_name))

    #print(caseDict)
    #sys.exit(1)
    tar_folder = "D:/Cathay_DB/Cathay_Image_Dataset_tmp"
    for case in caseDict.keys():
        case_folder = os.path.join(tar_folder, case)
        if not os.path.isdir(case_folder):
            os.mkdir(case_folder)
        if caseDict[case]["del"][0]:
            if caseDict[case]["del"][1] == 1:
                print("case:{} is {}".format(case, "貨車"))
                with open(os.path.join(case_folder, "貨車.txt"), 'w') as fid:
                    fid.write("")
            if caseDict[case]["del"][1] == 2:
                print("case:{} is {}".format(case, "廂型車"))
                with open(os.path.join(case_folder, "廂型車.txt"), 'w') as fid:
                    fid.write("")
            if caseDict[case]["del"][1] == 3:
                print("case:{} is {}".format(case, "機車"))
                with open(os.path.join(case_folder, "機車.txt"), 'w') as fid:
                    fid.write("")
            continue

        for img_path in caseDict[case]["imgs"]:
            basename = os.path.basename(img_path)
            copy_file = os.path.join(case_folder, basename)
            shutil.copy(img_path, copy_file)


def case_division(root):
    num_in_case = 5
    upper_folder = os.path.dirname(root)
    division_folder = os.path.join(upper_folder, "case_division")
    division_folder = cathay_utils.nt_path(division_folder)
    if not os.path.isdir(division_folder):
        try:
            os.makedirs(division_folder)
        except:
            logging.error("cannot create folder at path:{}".format(division_folder))
            return

    counter = 0
    case_counter = -1
    for rs, ds, fs in os.walk(root):
        if counter == 10:
            return

        lab1, lab2, folder = get_labs(rs)
        for i, f in enumerate(fs):
            if i == 0:
                case_counter += 1

            fpath = cathay_utils.path_join(folder, f)
            if 'json' in fpath:
                continue
            if 'txt' in fpath:
                continue
            file_name = os.path.basename(fpath).split(".jpg")[0]
            new_name = lab1+"_"+lab2+"_"+file_name+".jpg"
            #print(case_counter)
            sub_case_folder = division_folder + "/" + str(int(case_counter/num_in_case)) + "/" + lab1 + "_" + lab2
            #sub_case_folder = division_folder + "/" + str(int(case_counter/num_in_case))
            if not os.path.isdir(sub_case_folder):
                try:
                    os.makedirs(sub_case_folder)
                except:
                    logging.error("cannot create folder at path:{}".format(sub_case_folder))
                    sub_case_folder = division_folder

            new_path = os.path.join(sub_case_folder, new_name)
            logging.info("division {} to {}".format(fpath, new_path))
            shutil.copy(fpath, new_path)


def prepare_excel(classes, ):
    root = "./dataset_excel"
    if not os.path.isdir(root):
        os.makedirs(root)
    timestr = time.strftime("%Y-%m-%d", time.localtime())
    fname = os.path.join(root, timestr+'.xlsx')
    writer = pd.ExcelWriter(fname)

    writer_seq = COLS + classes
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


def create_dataloader(root):
    BATCHSIZE = 4
    DIM = 1024
    PAD = 32

    datasetRoot = root
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

    return dataloader, dataset

def app(root):
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
        di['PASS'] = ['N']
        di['TRAIN'] = ['N']
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
    args = parse_commands()
    # APP1
    #root = "D:/Cathay_DB/Cathay_Image_Dataset_tmp_02" #"D:/Cathay_DB/Cathay_tmp"
    #cathay_rename_folder(root)
    #cathay_labeled_parser(root)

    # APP2
    #root = "./data/case_division/"
    #app(root)

    # APP3
    #root = "D:/Cathay_DB/Cathay_Image_Training_Standard" #"./data/cathay_real"
    #case_division(root)
    #create_txt_table(root)



    # APP4
    # python datasetCheck.py --txt_table_path=D:/Cathay_DB/txts/cathay_txt_table_20201006.txt
    root = "D:/Cathay_DB/Cathay_Image_Database_Refined"
    update_txt_table(args, root)

    #load_labeled_data(args)