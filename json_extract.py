import os
import sys
import shutil
import json
import argparse

CARCLASS = ["CAF", "CAB", "CBF", "CBB", "CDFR", "CDFL", "CDBR", "CDBL", \
            "CFFR", "CFFL", "CFBR", "CFBL", "CS", "CMR", "CML", \
            "CLF", "CLB", "CWF", "CWB", "CG"]

def parse_commands():
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # default log folder
    # default case folder
    CASE_DIR = os.path.join(ROOT_DIR, "case")


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cathay file procession')

    parser.add_argument("--case_path", required=False,
                        default=CASE_DIR,
                        metavar="/path/to/case/",
                        help="'train' or 'evaluate' on MS COCO")

    parser.add_argument('--app', required=False,
                        default=0,
                        metavar="type of app",
                        help='type of app, 0: renaming, 1: result extract, 2: result division')

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

def file_rename(root):
    for rs, ds, fs in os.walk(root):
        if os.name == 'nt':
            folder = nt_path(rs)
        else:
            folder = rs
        #print(folder)
        if "tmp" in folder:
            continue
        #print(fs)
        for f in fs:
            fpath = path_join(folder, f)
            print(fpath)
            if 'jpg' in fpath:
                basename = os.path.basename(fpath)
                words = basename.split("_")
                new_name = ""
                for w in words:
                    if "理" in w:
                        continue
                    new_name = new_name+w+"_"
                new_name = new_name[:-1]
                print(new_name)
                new_path = path_join(folder, new_name)
                os.rename(fpath, new_path)

def extract_color_logo_plate(root, features=['color', 'logo', 'plate']):
    case_path = root
    ANS_Dict = {}
    for rs, ds, fs in os.walk(case_path):
        if os.name == 'nt':
            folder = nt_path(rs)
        else:
            folder = rs
        #print(folder)
        if "tmp" in folder:
            continue
        #print(fs)
        for f in fs:
            fpath = path_join(folder, f)
            print(fpath)
            if 'json' in fpath:
                basename = os.path.basename(fpath)
                words = (basename.split('.json')[0]).split('_')
                new_name = ""
                for w in words:
                    if "summary" in w:
                        continue
                    new_name = new_name + w + "_"
                new_name = new_name[:-1]

                ANS_Dict[new_name] = {}
                with open(fpath) as fid:
                    loadJson = json.load(fid)
                    if 'logo' in features:
                        ANS_Dict[new_name]['logo'] = loadJson['logo']
                    if 'color' in features:
                        ANS_Dict[new_name]['color'] = loadJson['color']
                    if 'plate' in features:
                        ANS_Dict[new_name]['plate'] = loadJson['plate']
                    if 'cars' in features:
                        carlist = loadJson['cars']
                        for c in CARCLASS:
                            ANS_Dict[new_name][c] = {'count':0, 'damages':[]}
                        for c in carlist:
                            lab = c['label']
                            if lab in ['CTA', 'CP']:
                                continue
                            ANS_Dict[new_name][lab]['count'] += 1

                #print(loadJson)
        #sys.exit(1)

    # write results
    with open('results_extract.txt', 'w') as fid:
        for i in sorted(ANS_Dict.keys()):
            wstr = str(i)+'\t'+ANS_Dict[i]['logo']+'\t'+ANS_Dict[i]['plate']+'\t'+ANS_Dict[i]['color']
            if 'cars' in features:
                for c in CARCLASS:
                    if ANS_Dict[i][c]['count'] == 0:
                        wstr = wstr+'\t'+'---'
                    else:
                        wstr = wstr+'\t'+str(ANS_Dict[i][c]['count'])
            wstr = wstr + '\n' + '\n'
            fid.write(wstr)

def case_division(root):
    case_path = root
    ANS_Dict = {}
    for rs, ds, fs in os.walk(case_path):
        if os.name == 'nt':
            folder = nt_path(rs)
        else:
            folder = rs
        # print(folder)
        if "tmp" in folder:
            continue

        for f in fs:
            fpath = path_join(folder, f)
            if 'predict' in fpath:
                continue
            elif 'json' in fpath:
                continue
            else:
                print("copy {}".format(fpath))
                words = os.path.basename(fpath).split('.')
                case_name = words[0]
                cur_folder = os.path.abspath(os.path.join(fpath, os.pardir))
                par_folder = os.path.abspath(os.path.join(cur_folder, os.pardir))
                org_img = fpath
                j_file = os.path.join(cur_folder, 'summary_'+words[0]+'.json')
                pred_img = os.path.join(cur_folder, 'predict_'+words[0]+'.jpg')
                if os.path.isfile(j_file) and os.path.isfile(pred_img):
                    #print("to par:{}, cur:{}, case:{}, json:{}, pred:{}".format(par_folder, cur_folder, case_name, j_file, pred_img))
                    div_folder = os.path.join(os.path.join(par_folder, 'div'), case_name)
                    new_j_file = os.path.join(div_folder, 'summary_'+words[0]+'.json')
                    new_pred_img = os.path.join(div_folder, 'predict_' + words[0] + '.jpg')
                    new_org_img = os.path.join(div_folder, case_name+'.jpg')
                    print("{}\n{}\n{}\n".format(new_org_img, new_pred_img, new_j_file))
                    if not os.path.isdir(os.path.join(par_folder, 'div')):
                        os.mkdir(os.path.join(par_folder, 'div'))
                    if not os.path.isdir(div_folder):
                        os.mkdir(div_folder)
                    shutil.copyfile(org_img, new_org_img)
                    shutil.copyfile(j_file, new_j_file)
                    shutil.copyfile(pred_img, new_pred_img)


if __name__ == '__main__':
    # python json_extract.py --case_path=./
    #case_path = './Cathay_20210126/phase1.1_1/01' #'./eval_100'
    args = parse_commands()
    #print(args)

    if int(args.app) == 0:
        file_rename(args.case_path)
    elif int(args.app) == 1:
        extract_features = ['color', 'logo', 'plate', 'cars']
        extract_color_logo_plate(args.case_path, extract_features)
    elif int(args.app) == 2:
        case_division(args.case_path)
    else:
        print("No app is selected")



    # file_rename(case_path)

    #extract_features = ['color', 'logo', 'plate', 'cars']
    #extract_color_logo_plate(case_path, extract_features)

    #case_division(case_path)