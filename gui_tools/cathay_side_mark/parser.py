import os
import sys
import argparse
import cv2
import shutil

import logging

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


def nt_path(root):
    nts = [p for p in root.split('\\')]
    path = nts[0]
    for p in nts[1:]:
        path = path + '/' + p
    return path

def path_join(p1, p2):
    if os.name == 'nt':
        path = p1 + '/' + p2
    else:
        path = os.path.join(p1, p2)
    return path

def get_labs(rs):
    if os.name == 'nt':
        folder = nt_path(rs)
    else:
        folder = rs
    logging.info("parsing folder: {}".format(folder))
    #print("parsing folder: {}".format(folder))
    folder_name = folder.split("/")[-1]
    lab1 = folder_name.split("_")[0]
    lab2 = folder_name.split("_")[-1]

    return lab1, lab2, folder


def img_labeled_parser(root, save2folder="side_db", startID=0):
    CLASS = {"00":"N", "10":"F", "20":"B", "01":"L", "02":"R", "11":"FL", "12":"FR", "21":"BL", "22":"BR"}
    DBDIR = path_join(nt_path(root), save2folder)
    if not os.path.isdir(DBDIR):
        try:
            os.makedirs(DBDIR)
        except:
            logging.error("cannot create folder at path:{}".format(DBDIR))

    fid_counter = startID
    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        if folder == DBDIR:
            continue

        for i, f in enumerate(fs):
            tpath = path_join(folder, f)
            if ".txt" in tpath:
                with open(tpath, 'r') as fid:
                    li = fid.readline()
                    rads = li.split(" ")
                    if len(rads) > 1:
                        code = rads[0]+rads[1]
                        print("file:{}, code:{}".format(tpath, code))
                        cls = CLASS[code]
                        # save to another folder
                        _name = tpath.split('.txt')[0]
                        img_path_org = _name + ".jpg"
                        new_img_path = path_join(DBDIR, cls + str(fid_counter) + ".jpg")
                        print("<find CP> img_org: {}, new: {}".format(img_path_org, new_img_path))
                        shutil.copyfile(img_path_org, new_img_path)
                        fid_counter += 1

    mark_path = path_join(DBDIR, str(fid_counter) + ".txt")
    os.close(os.open(mark_path, os.O_CREAT))


if __name__ == '__main__':
    # python parser.py --folder=./testdb
    args = parse_commands()

    img_labeled_parser(args.folder, startID=0)