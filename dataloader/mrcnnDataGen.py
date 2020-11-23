import os
import sys
import json
import time
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torchvision
import cv2
import skimage

from dataloader import img_utils

#from PIL import Image
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")

def getFrame(tif):
    frames = []
    for _f in tif.iter('Frame'):
        for _p in _f.iter('Page'):
            if _p.attrib['name'] != 'Default':
                frames.append(_f.attrib['name'])

    #print("frames: {}".format(frames))
    return frames

def path_join(p1, p2):
    if os.name == 'nt':
        path = p1 + '/' + p2
    else:
        path = os.path.join(p1, p2)
    return path

def ntPath(root):
    nts = [p for p in root.split('\\')]
    path = nts[0]
    for p in nts[1:]:
        path = path + '/' + p

    return path

def mask_of_page(ctr, height, width):
    from matplotlib.path import Path
    # mask of the page
    poly_verts = ctr.split(')')[:-1]
    poly_verts = [vert[1:] for vert in poly_verts]
    poly_verts = [(int(vert.split(',')[0]), int(vert.split(',')[-1])) for vert in poly_verts]
    #print('ctr:{} '.format(ctr))
    #print('poly_verts:{} '.format(poly_verts))

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T
    # print(len(points))

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((height, width))
    #print(grid.shape, grid)
    return grid


def getXMLPath(path, isZip=False, returnZref=False):
    """
    xml_path = getXMLPath("./dataset/car_damage_20200416")
    xml_path will be ./dataset/car_damage_20200416/configName/XMLs/car_damage_20200416/car_damage_20200416.xml
    """
    if isZip:
        zref = zipfile.ZipFile(path, "r")
        for tag in zref.namelist():
            if '/configName/XMLs' in tag:
                root = tag.split('/configName/XMLs')[0]
                case_name = root.split('/')[-1]
                if os.name == 'nt':
                    xmlPath = root+'/'+'configName/XMLs/'+case_name+'/'+case_name+'.xml'
                else:
                    xmlPath = os.path.join(root, 'configName', 'XMLs', case_name, case_name + '.xml')

                if xmlPath not in zref.namelist():
                    print('ERROR: Did not find {} in zip file'.format(xmlPath))
                    print('Error xml_path: {}'.format(xmlPath))
                    #print('Error xml_path: {}'.format(xmlPath), file=LOG)
                    #print('\ncase_name:{}'.format(case_name), file=ELOG)
                    #print('Error xml_path: {}'.format(xmlPath), file=ELOG)
                    zref.close()
                    return False, None
                else:
                    if returnZref:
                        return True, xmlPath, zref

                    zref.close()
                    return True, xmlPath

        zref.close()
        return False, None

    dirname = path.split('/')[-1]
    #print(dirname)
    xml_path = path+'/configName/XMLs/'+dirname+'/'+dirname+'.xml'
    if os.path.isfile(xml_path):
        return True, xml_path
    else:
        return False, None

def folderParser(root):
    datasetFolders = []
    for _root, dirs, files in os.walk(root):
        #print(_root, dirs, files)
        if 'configName' in dirs:
            if os.name == 'nt':
                datasetFolders.append((ntPath(_root), "folder"))
            else:
                datasetFolders.append((_root, "folder"))
        for f in files:
            if '.zip' in f:
                if os.name == 'nt':
                    root = ntPath(_root)
                    isPass, _ = getXMLPath(path_join(root, f), isZip=True)
                else:
                    isPass, _ = getXMLPath(path_join(root, f), isZip=True)
                if isPass:
                    datasetFolders.append((path_join(root, f), "zip"))

    return datasetFolders

def genImagePath(isZip, xml_path, root, tifName, name):
    if isZip:
        _t = xml_path.split('/configName/XMLs')[0]
        imgDir = path_join(path_join(_t, tifName), name)
        # print("caseName:{}, imgDir:{}".format(_t, imgDir))
        img_path = (root, imgDir)
    else:
        imgDir = path_join(root, tifName)
        img_path = path_join(imgDir, name)

    return img_path

def labelme(fpath, imgs, imgData, page_classes):
    with open(fpath) as fid:
        data = json.load(fid)
        #print(data)
        _path = data["imagePath"]
        root = os.path.dirname(fpath)
        img_path = path_join(root, _path)
        imgs.append(img_path)
        dataID = len(imgs) - 1
        imgData[dataID] = {'frame': {'rotation': "R0", \
                                     'lefttop_x': "0", \
                                     'lefttop_y': "0", \
                                     'width': data["imageWidth"], \
                                     'height': data["imageHeight"]}, \
                           'pages': []}
        shapes = data['shapes']
        #print(len(shapes), type(shapes))
        for _s in shapes:
            pname = _s['label']
            if pname in page_classes:
                points = _s['points']
                ctr = ""
                Xs, Ys = [], []
                for _p in points:
                    Xs.append(int(_p[0]))
                    Ys.append(int(_p[1]))
                    _ctr = "("+str(int(_p[0]))+", "+str(int(_p[1]))+")"
                    ctr += _ctr
                x1 = min(Xs)
                y1 = min(Ys)
                w = max(Xs) - x1
                h = max(Ys) - y1
                #print(ctr)
                imgData[dataID]['pages'].append({'name': pname, \
                                                 'bbox': str(x1)+" "+str(y1)+" "+str(w)+" "+str(h), \
                                                 'ctr': ctr})

        if len(imgData[dataID]['pages']) == 0:
            # remove this case
            imgs.pop()


def ITRILabelingPageField(root, xml_path, imgs, imgData,
                          page_classes,
                          field_classes,
                          isZip=[]):
    """
    data format: one image map to one page, image may be repeated
        imgs will be [(zipref, img_location), ...] or [img_location, ...]
        imgData will be {int: {'frame':{},
                               'page':{'name':'str', 'bbox':'str', 'ctr':'(10, 20)(30, 40)(50, 60)',
                                       'fields':[['ID', {attribs}], ...]} }}
    """
    if isZip[0]:
        zipRef = isZip[1]
        #print("zip namelist: {}".format(zipRef.namelist()))
        _tree = ET.parse(zipRef.open(xml_path))
    else:
        _tree = ET.parse(xml_path)

    _rootParse = _tree.getroot()
    for _tif in _rootParse.iter('Tif'):
        tifName = _tif.attrib['name']

        if tifName == 'configName':
            continue

        frames = getFrame(_tif)  # finding the labeled frame, filtering the default frame
        for _f in _tif.iter('Frame'):
            source = _f.attrib['source']
            name = _f.attrib['name']
            if name not in frames:
                continue

            img = genImagePath(isZip[0], xml_path, root, tifName, source)
            for _p in _f.iter('Page'):
                pname = _p.attrib['name'].split('_')[0]
                # print('page name: {}'.format(pname))

                if pname == 'Default':
                    continue

                elif pname in page_classes:
                    imgs.append(img)
                    dataID = len(imgs) - 1
                    imgData[dataID] = {'frame': {'rotation': _f.attrib['rotation'], \
                                                 'lefttop_x': _f.attrib['lefttop_x'], \
                                                 'lefttop_y': _f.attrib['lefttop_y'], \
                                                 'width': _f.attrib['width'], \
                                                 'height': _f.attrib['height']}, \
                                       'page': {'name': pname,\
                                                'bbox': _p.attrib['bbox'], \
                                                'ctr': _p.attrib['ctr'], \
                                                'fields': []}}

                    for field in field_classes:
                        for _field in _p.iter(field):
                            _fieldData = [field, {'lefttop_x': _field.attrib['lefttop_x'], \
                                                  'lefttop_y': _field.attrib['lefttop_y'], \
                                                  'width': _field.attrib['width'], \
                                                  'height': _field.attrib['height'], \
                                                  'labelName': _field.attrib['labelName'], \
                                                  'value': _field.attrib['value'], \
                                                  'font': _field.attrib['font']}]
                            # print(_fieldData)
                            imgData[dataID]['page']['fields'].append(_fieldData)

                    if len(imgData[dataID]['page']['fields']) == 0:
                        # remove this case
                        imgs.pop()
                else:
                    print('Error:Page name:{} can\'t be recognized'.format(pname))
                    # print('Error:Page name:{} can\'t be recognized'.format(pname), file=ELOG)
                    # print('\ncase_name:{}'.format(case), file=ELOG)
                    # print('tif:{}'.format(tif), file=ELOG)
                    # print('frame:{}, page:{} '.format(source, pname), file=ELOG)
                    # print('Error:Page name:{} can\'t be recognized'.format(pname), file=ELOG)
        if isZip[0]:
            zipRef.close()


def ITRILabelingPage(root, xml_path, imgs, imgData,
                     page_classes,
                     isZip=[]):
    """
    data format:
        imgs will be [(zipref, img_location), ...] or [img_location, ...]
        imgData will be {int: {'frame':{}, 'pages':[{'name':'str', 'bbox':'str','ctr':'(10, 20)(30, 40)(50, 60)}, ...] }}
    """
    if isZip[0]:
        zipRef = isZip[1]
        #print("zip namelist: {}".format(zipRef.namelist()))
        _tree = ET.parse(zipRef.open(xml_path))
    else:
        _tree = ET.parse(xml_path)

    _rootParse = _tree.getroot()
    for _tif in _rootParse.iter('Tif'):
        tifName = _tif.attrib['name']

        if tifName == 'configName':
            continue

        frames = getFrame(_tif)  # finding the labeled frame, filtering the default frame
        for _f in _tif.iter('Frame'):
            source = _f.attrib['source']
            name = _f.attrib['name']
            if name not in frames:
                continue

            img = genImagePath(isZip[0], xml_path, root, tifName, source)
            imgs.append(img)
            dataID = len(imgs) - 1
            imgData[dataID] = {'frame': {'rotation': _f.attrib['rotation'], \
                                         'lefttop_x': _f.attrib['lefttop_x'], \
                                         'lefttop_y': _f.attrib['lefttop_y'], \
                                         'width': _f.attrib['width'], \
                                         'height': _f.attrib['height']}, \
                               'pages': []}

            for _p in _f.iter('Page'):
                pname = _p.attrib['name'].split('_')[0]
                #print('page name: {}'.format(pname))

                itsfine = _p.findall('ITSFINE')

                if pname == 'Default':
                    continue
                elif pname in page_classes:
                    imgData[dataID]['pages'].append({'name': pname, \
                                                     'bbox': _p.attrib['bbox'], \
                                                     'ctr': _p.attrib['ctr']})
                else:
                    print('Error:Page name:{} can\'t be recognized'.format(pname))
                    #print('Error:Page name:{} can\'t be recognized'.format(pname), file=ELOG)
                    # print('\ncase_name:{}'.format(case), file=ELOG)
                    # print('tif:{}'.format(tif), file=ELOG)
                    #print('frame:{}, page:{} '.format(source, pname), file=ELOG)
                    #print('Error:Page name:{} can\'t be recognized'.format(pname), file=ELOG)

            if len(imgData[dataID]['pages']) == 0:
                # remove this case
                imgs.pop()

    if isZip[0]:
        zipRef.close()


class MRCnnDataset(object):
    def __init__(self,
                 root,
                 labelType='labelme',
                 resize_img=[False, 512, 512],
                 padding=[True, 32],
                 augmentation=None,
                 pil_process=True,
                 npy_process=False,
                 transforms=None,
                 page_classes=[],
                 field_classes=[],
                 data_type='page'):
        """
        :param root: the root dataset folder path
        :param resize_img: [True, min_dim, max_dim]
        :param transforms:
        :param page_classes:
        :param field_classes:
        :param data_type: 'page' or 'page_field', if 'page_field' should combine with 'page'
            the 'fields' are in the 'page'
        """
        self.labeltype = labelType
        self.page_classes = page_classes
        self.field_classes = field_classes
        self.data_type = data_type

        """
        timestr = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        ERRDIR = os.path.join(root, timestr)
        if not os.path.isdir(ERRDIR):
            os.makedirs(ERRDIR)
        ELOG = open(os.path.join(ERRDIR, 'Elogs.txt'), 'w')
        """
        self.is_resize = resize_img[0]
        self.min_dim = resize_img[1]
        self.max_dim = resize_img[2]
        self.padding = padding
        self.augmentation = augmentation
        self.pil = pil_process
        self.npy = npy_process
        self.transforms = transforms

        self.imgs = []
        self.imgData = {}
        # self.imgs = [(), (), ...]
        # self.imgData = {0: {'frame': {'rotation':'R0', 'lefttop_x':'0', 'lefttop_y':'0', 'width':'128',\
        #                               'height':'128'},
        #                     'pages':[{'name':'abc', 'bbox':'x1, y1, x2, y2',\
        #                               'ctr':'(11, 14)(16, 11)(12, 14)(16, 19)'}, {}, ...] }, 1:{}, ...}

        if self.labeltype == 'labelme':
            for _root, dirs, files in os.walk(root):
                if os.name == 'nt':
                    folder = ntPath(_root)
                else:
                    folder = _root
                for f in files:
                    if '.json' in f:
                        fpath = path_join(folder, f)
                        print("fpath: {}".format(fpath))
                        labelme(fpath, self.imgs, self.imgData,
                                page_classes=self.page_classes)

        elif self.labeltype == 'ITRILabel':
            datasets = folderParser(root)
            # print("datasets: {}".format(datasets))

            for _i, db in enumerate(datasets):
                print("parsering db:{}".format(db[0]))
                isZip = True if db[1] == 'zip' else False
                if isZip:
                    is_pass, _xml_path, _zref = getXMLPath(db[0], isZip=True, returnZref=True)
                    print("  xml_path: {}".format(_xml_path))
                    #is_pass = False
                    if is_pass:
                        if self.data_type == 'page':
                            # ["b0090101", "b0070301", "B0080301", "B0070101"]
                            ITRILabelingPage(db[0], _xml_path, self.imgs, self.imgData,
                                             page_classes=self.page_classes,
                                             isZip=[True, _zref])
                        elif self.data_type == 'page_field':
                            ITRILabelingPageField(db[0], _xml_path, self.imgs, self.imgData,
                                                  page_classes=self.page_classes,
                                                  field_classes=self.field_classes,
                                                  isZip=[True, _zref])
                        else:
                            pass
                else:
                    is_pass, _xml_path = getXMLPath(db[0])
                    print("  xml_path: {}".format(_xml_path))
                    if is_pass:
                        if self.data_type == 'page':
                            ITRILabelingPage(db[0], _xml_path, self.imgs, self.imgData,
                                             self.page_classes,
                                             isZip=[False])
                        elif self.data_type == 'page_field':
                            ITRILabelingPageField(db[0], _xml_path, self.imgs, self.imgData,
                                                  page_classes=self.page_classes,
                                                  field_classes=self.field_classes,
                                                  isZip=[False])
                        else:
                            pass
        else:
            pass

        #print(self.imgs)
        #print(self.imgData)

    def __getitem__(self, idx):
        # load images ad masks
        _tmp = self.imgs[idx]
        _data = self.imgData[idx]
        #print(_tmp)
        if type(_tmp) is tuple:
            zref = zipfile.ZipFile(_tmp[0], "r")
            data = zref.read(_tmp[1])
            img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            if img.shape[2] != 3:
                img = img_utils.gray_3_ch(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            zref.close()
        else:
            #print("idx: {}, name: {}".format(idx, _tmp))
            img = cv2.imread(_tmp)
            #print(img.shape, img.ndim)
            #sys.exit(1)
            if img.shape[2] != 3:
                img = img_utils.gray_3_ch(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = {}
        print(_data, idx)

        height = int(_data['frame']['height'])
        width = int(_data['frame']['width'])
        rotation = _data['frame']['rotation']

        if rotation == 'R90':
            img = np.rot90(img, 3)
        elif rotation == 'R180':
            img = np.rot90(img, 2)
        elif rotation == 'R270':
            img = np.rot90(img, 1)
        elif rotation == 'R0':
            img = img
        else:
            print('Error Rotation:{} '.format(rotation))

        boxes = []
        if self.data_type == 'page':
            if self.padding[0]:
                padding = [(self.padding[1], self.padding[1]), (self.padding[1], self.padding[1]), (0, 0)]
                img = np.pad(img, padding, mode='constant', constant_values=0)

            if self.is_resize:
                img, window, scale, resize_padding, crop = img_utils.resize_image(img, min_dim=self.min_dim,
                                                                                  max_dim=self.max_dim)
            img = img.astype(np.uint8)
            if self.augmentation:
                img, det = img_utils.img_augmentation(img, self.augmentation)
                #print("data augmentation: {}".format(det))

            num_objs = len(_data['pages'])
            labels = np.zeros((num_objs,))
            #masks = np.zeros((num_objs, height, width))
            masks = np.zeros((num_objs, img.shape[0], img.shape[1]))
            for _i, _p in enumerate(_data['pages']):
                labels[_i] = self.page_classes.index(_p['name'])+1
                mask_org = mask_of_page(_p['ctr'], height, width)
                mask_img = np.zeros((height, width, 3))
                mask_img[:, :, 0] = mask_org * 1.0
                mask_img[:, :, 1] = mask_org * 1.0
                mask_img[:, :, 2] = mask_org * 1.0
                if self.padding[0]:
                    mask_img = np.pad(mask_img, padding, mode='constant', constant_values=0)

                if self.is_resize:
                    mask_img = img_utils.resize_mask(mask_img, scale, resize_padding, crop)

                mask_img = mask_img.astype(np.uint8)
                if self.augmentation:
                    mask_img = img_utils.mask_augmentation(img, mask_img, det)

                #print(img.shape, mask_img.shape)
                #sys.exit(1)

                mask_img[np.where(mask_img > 0)] = 1.0
                mask = mask_img[:, :, 0]
                masks[_i, :, :] = mask
                #print(masks[_i, :, :]*255)
                #sys.exit(1)
                pos = np.where(mask > 0)
                #print(pos[0].shape, pos[1].shape)
                try:
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    print(xmin, xmax, ymin, ymax)
                    boxes.append([xmin, ymin, xmax, ymax])
                except:
                    print("WARNING: pos shape error, skip this case {},{}".format(pos[0].shape, pos[1].shape))
                    continue
                #print(type(mask))
            #print('labels: {}, boxes: {}'.format(labels, boxes))
        elif self.data_type == 'page_field':
            page = mask_of_page(_data['page']['ctr'], height, width)
            pos = np.where(page)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # === update img to the page size ==== #
            img = img[ymin:ymax, xmin:xmax, :]

            if self.padding[0]:
                padding = [(self.padding[1], self.padding[1]), (self.padding[1], self.padding[1]), (0, 0)]
                img = np.pad(img, padding, mode='constant', constant_values=0)

            if self.is_resize:
                img, window, scale, resize_padding, crop = img_utils.resize_image(img, min_dim=self.min_dim,
                                                                                  max_dim=self.max_dim)
            img = img.astype(np.uint8)
            if self.augmentation:
                img, det = img_utils.img_augmentation(img, self.augmentation)
                print("data augmentation: {}".format(det))

            num_objs = len(_data['page']['fields'])
            #print("num_objs:{}".format(num_objs))
            labels = np.zeros((num_objs,))
            # masks: (N, height, width)
            #masks = np.zeros((num_objs, ymax-ymin, xmax-xmin))
            masks = np.zeros((num_objs, img.shape[0], img.shape[1]))
            for _i, _p in enumerate(_data['page']['fields']):
                labels[_i] = self.field_classes.index(_p[0])+1
                fxmin = int(_p[1]['lefttop_x']) if int(_p[1]['lefttop_x']) > xmin else xmin
                fxmax = fxmin+int(_p[1]['width'])
                fxmax = fxmax if fxmax < xmax else xmax
                fymin = int(_p[1]['lefttop_y']) if int(_p[1]['lefttop_y']) > ymin else ymin
                fymax = fymin+int(_p[1]['height'])
                fymax = fymax if fymax < ymax else ymax

                mask_img = np.zeros((ymax-ymin, xmax-xmin, 1))
                mask_img[(fymin-ymin):(fymax-ymin), (fxmin-xmin):(fxmax-xmin), 0] = 1.0

                if self.padding[0]:
                    mask_img = np.pad(mask_img, padding, mode='constant', constant_values=0)

                if self.is_resize:
                    mask_img = img_utils.resize_mask(mask_img, scale, resize_padding, crop)

                mask_img = mask_img.astype(np.uint8)
                if self.augmentation:
                    mask_img = img_utils.mask_augmentation(img, mask_img, det)

                mask = mask_img[:, :, 0]
                masks[_i, :, :] = mask
                pos = np.where(mask > 0)
                fxmin = np.min(pos[1])
                fxmax = np.max(pos[1])
                fymin = np.min(pos[0])
                fymax = np.max(pos[0])
                boxes.append([fxmin, fymin, fxmax, fymax])
        else:
            pass

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        #print(boxes, _data, _tmp, idx,)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.pil:
            #img = torchvision.transforms.ToPILImage()(img)
            #img = torchvision.transforms.ToPILImage()(img).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img.copy())
            #img = torchvision.transforms.ToTensor()(img)

        if self.npy:
            # with 0~255 value, training may have loss in none
            if self.transforms is not None:
                for tr in self.transforms:
                    img = tr[0](img)
            img = img.transpose((2, 0, 1))    # numpy as (H, W, C) to pytorch as (C, H, W)
            img = torch.from_numpy(img.copy()).float()

        return img, target

    def __len__(self):
        return len(self.imgs)
