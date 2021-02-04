import os
import numpy as np
import webcolors
import argparse
import random
import colorsys

COLORWEIGHT = {0:{'loc':(0, 0, 341, 341), 'w':0.5}, 1:{'loc':(341, 0, 682, 341), 'w':1.0}, 2:{'loc':(682, 0, 1024, 341), 'w':0.5}, \
               3:{'loc':(0, 341, 341, 682), 'w':1.0}, 4:{'loc':(341, 341, 682, 682), 'w':1.0}, 5:{'loc':(682, 341, 1024, 682), 'w':1.0}, \
               6:{'loc':(0, 682, 341, 1024), 'w':0.5}, 7:{'loc':(341, 682, 682, 1024), 'w':1.0}, 8:{'loc':(682, 682, 1024, 1024), 'w':0.5} }

COLORSYS = {'#000000':'black', '#141414':'black', '#2d2d2d':'black', '#464646':'black',\
            '#ffffff':'white', '#f1f1f1':'white', '#e6e6e6':'white', '#c8c8c8':'white', '#b4c8c8':'white', \
            '#c0c0c0':'white', '#bebebe':'white',\
            '#828282':'silver', '#878787':'silver', '#919191':'silver',\
            '#9b9b9b':'silver', '#a3a3a3':'silver', '#aaaaaa':'silver', '#b0b0b0':'silver', '#b9b9b9':'silver',\
            '#505050':'gray', '#5f5f5f':'gray', '#696969':'gray', '#6e6e6e':'gray', '#737373':'gray', '#787878':'gray', \
            '#ff0000':'red', '#e60000':'red', '#ff2400':'red', '#ff3d00':'red', '#e32636':'red', '#b80007':'red', '#872300':'red', \
            '#b22222':'red', '#ff4500':'red', '#8b0000':'red', '#783c3c':'red',\
            '#0000ff':'blue', '#007FFF':'blue', '#6495ed':'blue', '#003399':'blue', '#000080':'blue', '#003366':'blue', \
            '#1e90ff':'blue', '#0d33ff':'blue', '#002fa7':'blue', '#4d80e6':'blue', '#3200ff':'blue', '#3c5a78':'blue', \
            '#5078a0':'blue', '#284664':'blue', '#96c8c8':'blue',\
            '#008000':'green', '#00ff00':'green', '#006400':'green', '#00ff80':'green', '#4de680':'green', \
            '#2e8b57':'green', '#3cb371':'green', '#127436':'green', '#2e8b57':'green', '#78a078':'green', '#96aa96':'green',\
            '#ffff00':'yellow', '#ffff50':'yellow', '#ffffA0':'yellow', '#ffd600':'yellow', '#ffd680':'yellow', '#d9b200':'yellow', \
            '#d9b250':'yellow', '#ffcc00':'yellow', '#ffef00':'yellow', '#ffff4d':'yellow'}

CARCOLORMAP = {'CAF':(1.0,0.0,0.0), 'CAB':(1.0,0.0,0.0), 'CBF':(0.0,0.0,1.0), 'CBB':(0.0,0.0,1.0), \
               'CDFR':(0.0,1.0,1.0), 'CDFL':(0.0,1.0,1.0), 'CDBR':(0.0,1.0,0.0), 'CDBL':(0.0,1.0,0.0), \
               'CFFR':(1.0,0.0,1.0), 'CFFL':(1.0,0.0,1.0), 'CFBR':(0.5,0.0,0.0), 'CFBL':(0.5,0.0,0.0), \
               'CS':(1.0,0.5,1.0), 'CMR':(0.5,1.0,0.0), 'CML':(0.5,1.0,0.0), 'CG':(1.0,1.0,0.0), \
               'CLF':(1.0,0.0,0.5), 'CLB':(0.5,0.0,1.0), 'CWF':(1.0,1.0,0.5), 'CWB':(1.0,1.0,0.5)}

DCOLORMAP = {'DS':(1.0, 0.0, 0.0), 'DD':(0.0, 1.0, 0.0), 'DC':(0.0, 0.0, 1.0)}

def bgr2hsv(rgb):
    hsv = colorsys.rgb_to_hsv(rgb[2]/255.0, rgb[1]/255.0, rgb[0]/255.0)
    #print("hsv={}".format(hsv))
    return hsv

def car_color_map(lab):
    return CARCOLORMAP[lab]

def damage_color_map(lab):
    return DCOLORMAP[lab]

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
    print(colors)
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

def color_weight(loc):
    x1, y1, x2, y2 = loc
    #print(x1, y1, x2, y2)
    iou = [0, 0.0]
    for i in COLORWEIGHT.keys():
        target = COLORWEIGHT[i]['loc']
        _iou = bbox_iou(loc, target)
        #print(_iou)
        if _iou > iou[1]:
            iou = [i, _iou]

    #print("color iou:{}".format(iou))
    #sys.exit(1)
    weight = COLORWEIGHT[iou[0]]['w']

    return weight

def closest_colour(requested_colour, hsv):
    min_colours = {}
    # webcolors.HTML4_HEX_TO_NAMES.items()
    for key, name in COLORSYS.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    #print(min_colours)
    color = min_colours[min(min_colours.keys())]
    #if color == 'silver':
    #    if hsv[2] > 0.7:
    #        color = 'white'

    return color #min_colours[min(min_colours.keys())]

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

    parser.add_argument('--damage_on', required=False,
                        default=False,
                        metavar="<True|False>",
                        help="open damage segmentation")

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