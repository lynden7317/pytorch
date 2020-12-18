import os
import sys
import json
import numpy as np
import cv2
import argparse
import logging

from matplotlib.path import Path

import cathay_utils

import shutil

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

    parser.add_argument('--extract_type', required=False,
                        default="CTA",
                        metavar="CTA",
                        help="car portions")

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

def filter_CP_and_rename(root, save2folder="CPs"):
    CPDIR = cathay_utils.path_join(cathay_utils.nt_path(root), save2folder)
    if not os.path.isdir(CPDIR):
        try:
            os.makedirs(CPDIR)
        except:
            logging.error("cannot create folder at path:{}".format(CPDIR))

    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        for i, f in enumerate(fs):
            fpath = cathay_utils.path_join(folder, f)
            #print(fpath)
            if ".json" in fpath:
                with open(fpath) as fid:
                    jfile = json.load(fid)
                    shapes = jfile['shapes']
                    for _s in shapes:
                        if _s["label"] == "CP":
                            #print("CP: {}".format(fpath))
                            # save to another folder
                            _name = fpath.split('.json')[0]
                            fname = os.path.dirname(_name).split('/')[-1]
                            img_name = os.path.basename(_name)
                            img_names = img_name.split('_')
                            if len(img_names) > 0:
                                img_name = img_names[-1]
                            else:
                                img_name = img_name
                            #print("img_name: {}, f: {}".format(img_name, fname))
                            img_path_org = _name + ".jpg"
                            new_img_path = cathay_utils.path_join(CPDIR, fname+"_"+img_name+".jpg")
                            print("<find CP> img_org: {}, new: {}".format(img_path_org, new_img_path))
                            shutil.copyfile(img_path_org, new_img_path)

def extract_tar_2_folder(root, extract_tar="CTA", tag="", save2folder="CTA"):
    TARDIR = cathay_utils.path_join("./", save2folder+"DIR")  # cathay_utils.nt_path(root)
    if not os.path.isdir(TARDIR):
        try:
            os.makedirs(TARDIR)
        except:
            logging.error("cannot create folder at path:{}".format(TARDIR))

    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        for i, f in enumerate(fs):
            fpath = cathay_utils.path_join(folder, f)
            print(fpath)
            if ".json" in fpath:
                with open(fpath) as fid:
                    jfile = json.load(fid)
                    shapes = jfile['shapes']
                    for _s in shapes:
                        counter = 0
                        if _s["label"] == extract_tar:
                            points = np.array(_s["points"])
                            #print(points, points.shape)
                            bx1, by1, bx2, by2 = int(np.min(points[:,0], axis=0)), int(np.min(points[:,1], axis=0)), \
                                                 int(np.max(points[:, 0], axis=0)), int(np.max(points[:,1], axis=0))
                            if bx1 < 0:
                                bx1 = 0
                            if by1 < 0:
                                by1 = 0
                            #print(bx1, by1, bx2, by2)
                            _name = fpath.split('.json')[0]
                            img_path_org = _name + ".jpg"
                            fname = os.path.dirname(_name).split('/')[-1]
                            img_name = os.path.basename(_name)
                            img_names = img_name.split('_')
                            if len(img_names) > 0:
                                img_name = img_names[-1]
                            else:
                                img_name = img_name
                            new_img_path = cathay_utils.path_join(TARDIR, tag+"_"+fname+"_"+img_name+"_"+str(counter)+".jpg")

                            try:
                                img = cv2.imread(img_path_org)
                                crop_img = img[by1:by2, bx1:bx2, :]
                                cv2.imwrite(new_img_path, crop_img)
                            except:
                                print(by1, by2, bx1, bx2)
                                print("cannot save crop_img: {} to {}".format(img_path_org, new_img_path))
                            counter += 1

def extract_mask_2_folder(root, extract_tar="CTA", tag="", save2folder="CTA"):
    TARDIR = cathay_utils.path_join("./", save2folder + "DIR")
    if not os.path.isdir(TARDIR):
        try:
            os.makedirs(TARDIR)
        except:
            logging.error("cannot create folder at path:{}".format(TARDIR))

    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        for i, f in enumerate(fs):
            fpath = cathay_utils.path_join(folder, f)
            print(fpath)
            if ".json" in fpath:
                with open(fpath) as fid:
                    jfile = json.load(fid)
                    shapes = jfile['shapes']
                    
                    _name = fpath.split('.json')[0]
                    img_path_org = _name + ".jpg"
                    fname = os.path.dirname(_name).split('/')[-1]
                    img_name = os.path.basename(_name)
                    img_names = img_name.split('_')
                    if len(img_names) > 0:
                        img_name = img_names[-1]
                    else:
                        img_name = img_name
                    
                    counter = 0
                    img = cv2.imread(img_path_org)
                    print("shape: {}".format(img.shape))
                    for _s in shapes:    
                        if _s["label"] == extract_tar:
                            print(_s["label"], counter)
                            points = np.array(_s["points"])
                            bx1, by1, bx2, by2 = int(np.min(points[:, 0], axis=0)), int(np.min(points[:, 1], axis=0)), \
                                                 int(np.max(points[:, 0], axis=0)), int(np.max(points[:, 1], axis=0))
                            
                            if bx1 < 0:
                                bx1 = 0
                            if by1 < 0:
                                by1 = 0
                            
                            new_img_path = cathay_utils.path_join(TARDIR, tag+"_"+fname+"_"+img_name+"_"+str(counter)+".jpg")

                            try:
                                mask_img = np.zeros(img.shape)
                                ppath = Path(points)
                                
                                #print(ppath)

                                nx, ny = img.shape[1], img.shape[0]
                                #print("shape: {}, x:{}, y:{}".format(img.shape, nx, ny))
                                
                                xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
                                xs, ys = xs.flatten(), ys.flatten()
                                grid_points = np.vstack((xs, ys)).T
                                #print(grid_points.shape)

                                grid = ppath.contains_points(grid_points)
                                grid = grid.reshape((ny, nx))
                                rows, cols = np.where(grid == True)
                                #print(rows.shape, cols.shape)
                                #print(ppath, grid.shape)

                                for _c in range(3):
                                    mask_img[rows, cols, _c] = img[rows, cols, _c]

                                mask_img = mask_img[by1:by2, bx1:bx2]

                                cv2.imwrite(new_img_path, mask_img)
                                #cv2.imwrite(new_img_path+'.jpg', img[by1:by2, bx1:bx2])
                            except:
                                print(by1, by2, bx1, bx2)
                                print("cannot save crop_img: {} to {}".format(img_path_org, new_img_path))
                            
                            counter += 1
                            #sys.exit(1)

def remove_empty_folder(root):
    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        if os.path.exists(folder):
            if len(os.listdir(folder)) == 0:
                print("empty folder: {}, remove".format(folder))
                os.rmdir(folder)

def remove_no_labeled_jpg(root):
    for rs, ds, fs in os.walk(root):
        lab1, lab2, folder = get_labs(rs)
        for i, f in enumerate(fs):
            fpath = cathay_utils.path_join(folder, f)
            print(fpath)
            if ".jpg" in fpath:
                js_remove = False
                json_file = fpath.split('.jpg')[0]+'.json'
                if os.path.isfile(json_file):
                    with open(json_file) as fid:
                        jfile = json.load(fid)
                        shapes = jfile['shapes']
                        if len(shapes) == 0:
                            print("json file no content, remove")
                            os.remove(fpath)
                            js_remove = True
                            
                    if js_remove:
                        os.remove(json_file)
                else:
                    print("remove this jpg")
                    os.remove(fpath)

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

    #check_json_polys(args.folder)
    #refine_json(args.folder)
    #filter_CP_and_rename(args.folder)

    # python polyCheck.py --folder=D:/Cathay_DB/Ly/0/0001-BA_1516SR01650 --tag="D" --extract_type="CTA"/"CP"
    #extract_tar_2_folder(args.folder, args.extract_type, args.tag, args.extract_type)
    
    """
    CLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR", "CDBL", \
             "CFFR", "CFFL", "CFBR", "CFBL", "CS", "CMR", "CML", \
             "CLF", "CLB", "CWF", "CWB", "CG", "CTA", "CP"]
    
    #CLASS = ["CAF", "CAB"]
    for _c in CLASS:
        extract_mask_2_folder(args.folder, _c, args.tag, _c)
    """
    
    remove_no_labeled_jpg(args.folder)
    
    