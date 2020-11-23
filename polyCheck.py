import os
import sys
import json
import argparse
import logging

import cathay_utils

def parse_commands():
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # default log folder
    MODEL_DIR = os.path.join(ROOT_DIR, "log")
    # default case folder
    CASE_DIR = os.path.join(ROOT_DIR, "case")


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cathay Car Damage Dataset.')

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

def check_json_polys(root):
    # read txt table
    CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR", "CDBL", \
             "CFFR", "CFFL", "CFBR", "CFBL", "CS", "CMR", "CML", \
             "CLF", "CLB", "CWF", "CWB", "CG", "CTA", "CP"]
    CDict = dict.fromkeys(CLASS, 0)
    TOTAL = 0
    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        for i, f in enumerate(fs):
            fpath = cathay_utils.path_join(folder, f)
            print(fpath)
            if ".json" in fpath:
                with open(fpath) as fid:
                    jfile = json.load(fid)
                    #print(jfile)
                    shapes = jfile['shapes']
                    #print(len(shapes))
                    TOTAL += len(shapes)
                    for _s in range(len(shapes)):
                        CDict[shapes[_s]["label"]] += 1
                    #sys.exit(1)

    print(CDict, TOTAL)

def refine_json(root):
    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        for i, f in enumerate(fs):
            fpath = cathay_utils.path_join(folder, f)
            print(fpath)
            if ".json" in fpath:
                with open(fpath) as fid:
                    jfile = json.load(fid)
                    imgpath = jfile['imagePath']
                    jfile['imagePath'] = imgpath.split('/')[-1]

                print("save labelme path:{}".format(fpath))
                json_data = json.dumps(jfile, indent=2, separators=(',', ': '))
                with open(fpath, 'w') as fid:
                    fid.write(json_data)



if __name__ == '__main__':
    # python polyCheck.py --folder=D:/Cathay_DB/Ly/0/0001-BA_1516SR01650
    args = parse_commands()

    check_json_polys(args.folder)
    #refine_json(args.folder)