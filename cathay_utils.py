import os
import numpy as np
import webcolors
import argparse
import random
import colorsys

COLORSYS = {'#000000':'black', \
            '#ffffff':'white', \
            '#c0c0c0':'silver', \
            '#808080':'gray', \
            '#ff0000':'red', '#e60000':'red', '#ff2400':'red', '#ff3d00':'red', '#e32636':'red', '#b80007':'red', '#872300':'red', \
            '#b22222':'red', '#ff4500':'red', '#8b0000':'red',\
            '#0000ff':'blue', '#007FFF':'blue', '#6495ed':'blue', '#003399':'blue', '#000080':'blue', '#003366':'blue', \
            '#1e90ff':'blue', '#0d33ff':'blue', '#002fa7':'blue', '#4d80e6':'blue',\
            '#008000':'green', '#ccff00':'green', '#00ff00':'green', '#006400':'green', '#00ff80':'green', '#4de680':'green', \
            '#2e8b57':'green', '#3cb371':'green', '#127436':'green', '#2e8b57':'green',\
            '#ffff00':'yellow', '#ffff50':'yellow', '#ffffA0':'yellow', '#ffd600':'yellow', '#ffd680':'yellow', '#d9b200':'yellow', \
            '#d9b250':'yellow', '#ffcc00':'yellow', '#ffef00':'yellow', '#ffff4d':'yellow'}

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

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

def closest_colour(requested_colour):
    min_colours = {}
    # webcolors.HTML4_HEX_TO_NAMES.items()
    for key, name in COLORSYS.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    #print(min_colours)
    return min_colours[min(min_colours.keys())]

def parse_commands():
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # default log folder
    MODEL_DIR = os.path.join(ROOT_DIR, "log")
    # default case folder
    CASE_DIR = os.path.join(ROOT_DIR, "case")


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cathay Car Damage Detection.')

    parser.add_argument("--case_path", required=False,
                        default=CASE_DIR,
                        metavar="/path/to/case/",
                        help="'train' or 'evaluate' on MS COCO")

    parser.add_argument('--log_path', required=False,
                        default=MODEL_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    parser.add_argument('--plot_on', required=False,
                        default=False,
                        metavar="<True|False>",
                        help="open plot image function")

    parser.add_argument('--mode', required=False,
                        default='train',
                        metavar="train/eval",
                        help='train/eval')

    parser.add_argument('--case_mode', required=False,
                        default='single',
                        metavar="single/multiple",
                        help="single/multiple")

    args = parser.parse_args()

    return args

def bbox_iou(bA, bB):
    # bbox = [xmin, ymin, xmax, ymax]
    xA = max(bA[0], bB[0])
    yA = max(bA[1], bB[1])
    xB = min(bA[2], bB[2])
    yB = min(bA[3], bB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (bA[2] - bA[0] + 1) * (bA[3] - bA[1] + 1)
    boxBArea = (bB[2] - bB[0] + 1) * (bB[3] - bB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def cal_iou(tar, comp, org_shape):
    merge = False
    t_pos = np.zeros((org_shape[0], org_shape[1]))
    c_pos = np.zeros((org_shape[0], org_shape[1]))
    t_cols, t_rows = tar[4]
    c_cols, c_rows = comp[4]
    t_pos[t_cols, t_rows] = 1
    c_pos[c_cols, c_rows] = 1
    iou_pos = t_pos * c_pos
    iou = np.sum(np.sum(iou_pos, axis=0), axis=0)
    print(len(t_cols), len(c_cols))
    print("iou:{}".format(iou))
    intersection = iou/len(c_cols)
    if intersection > 0.7:
        print("merge to target: {}".format(intersection))
        merge = True

    return merge

#requested_colour = (122, 105, 94)
#name = closest_colour(requested_colour)
#print(name)