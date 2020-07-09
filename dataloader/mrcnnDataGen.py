import os
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torchvision
import cv2

#from PIL import Image

def collate_fn(batch):
    return tuple(zip(*batch))

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
    #print(grid.shape)
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

            img = genImagePath(isZip[0], xml_path, root, tifName, name)
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

            img = genImagePath(isZip[0], xml_path, root, tifName, name)
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


class MRCnnXMLDataset(object):
    def __init__(self, root,
                 transforms=None,
                 page_classes=[],
                 field_classes=[],
                 data_type='page'):
        """
        :param root: the root dataset folder path
        :param transforms:
        :param page_classes:
        :param field_classes:
        :param data_type: 'page' or 'page_field', if 'page_field' should combine with 'page'
            the 'fields' are in the 'page'
        """

        self.datasets = folderParser(root)
        #print("datasets: {}".format(self.datasets))

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

        self.transforms = transforms
        self.imgs = []
        self.imgData = {}

        for _i, db in enumerate(self.datasets):
            print("parsering db:{}".format(db[0]))
            isZip = True if db[1] == 'zip' else False
            if isZip:
                is_pass, _xml_path, _zref = getXMLPath(db[0], isZip=True, returnZref=True)
                print("  xml_path: {}".format(_xml_path))
                #is_pass = False
                if is_pass:
                    if self.data_type == 'page':
                        ITRILabelingPage(db[0], _xml_path, self.imgs, self.imgData,
                                         page_classes=["b0090101", "b0070301", "B0080301", "B0070101"],
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

        #print(self.imgs)
        #print(self.imgData)

    def __getitem__(self, idx):
        # load images ad masks
        _tmp = self.imgs[idx]
        _data = self.imgData[idx]
        if type(_tmp) is tuple:
            zref = zipfile.ZipFile(_tmp[0], "r")
            data = zref.read(_tmp[1])
            img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            zref.close()
        else:
            img = cv2.imread(_tmp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = {}
        #print(_data, idx)

        height = int(_data['frame']['height'])
        width = int(_data['frame']['width'])
        boxes = []
        if self.data_type == 'page':
            num_objs = len(_data['pages'])
            labels = np.zeros((num_objs,))
            masks = np.zeros((num_objs, height, width))
            for _i, _p in enumerate(_data['pages']):
                labels[_i] = self.page_classes.index(_p['name'])+1
                mask = mask_of_page(_p['ctr'], height, width)
                masks[_i, :, :] = mask
                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
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

            num_objs = len(_data['page']['fields'])
            #print("num_objs:{}".format(num_objs))
            labels = np.zeros((num_objs,))
            # masks: (N, height, width)
            masks = np.zeros((num_objs, ymax-ymin, xmax-xmin))
            for _i, _p in enumerate(_data['page']['fields']):
                labels[_i] = self.field_classes.index(_p[0])+1
                fxmin = int(_p[1]['lefttop_x']) if int(_p[1]['lefttop_x']) > xmin else xmin
                fxmax = fxmin+int(_p[1]['width'])
                fxmax = fxmax if fxmax < xmax else xmax
                fymin = int(_p[1]['lefttop_y']) if int(_p[1]['lefttop_y']) > ymin else ymin
                fymax = fymin+int(_p[1]['height'])
                fymax = fymax if fymax < ymax else ymax
                masks[_i, (fymin-ymin):(fymax-ymin), (fxmin-xmin):(fxmax-xmin)] = 1
                boxes.append([fxmin-xmin, fymin-ymin, fxmax-xmin, fymax-ymin])
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

        img = torchvision.transforms.ToPILImage()(img).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        img = torchvision.transforms.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


if __name__ == '__main__':
    """
    usage:
      prepared attributes:
        dataset folder: ex "./dataset/test"
        page_classes: ex ['name1', ...]
        field_classes: ex ['name1', ...]
        data_type = 'page' or 'page_field'
    """
    page_classes = ['AA001A', 'AA001B', 'AA002A', 'AA003B', 'AA004A', 'AA005B', 'ANA001A', 'ANA001B', 'ANA002A', 'ANA003A', 'CA001A', 'CA001B', 'CD001A', 'CD001B', 'CD002B', 'CD003A', 'CI001A', 'CI001B', 'CP001A', 'CP001B', 'CP002B', 'EM001A', 'EM001B', 'EVA001A', 'EVA001B', 'EVA002A', 'EVA002B', 'JAL001A', 'JAL001B', 'JAL002A', 'JE001A', 'JE001B', 'JE002B', 'JE003B', 'JS001A', 'JS002A', 'JS003A', 'MD001B', 'PA001A', 'PA002A', 'PA002B', 'SC001A', 'SC001B', 'TH001A', 'TH001B', 'TH002A', 'TH002B', 'TIG001B', 'TIG002A', 'TIG002B', 'XI001A', 'XI001B']
    field_classes = ['Name', 'Date', 'Flight', 'FromCity', 'ToCity', 'FromId', 'ToId']

    import mask_rcnn

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.freeze_support()

    dataset = MRCnnXMLDataset("./dataset/test2",
                              page_classes=page_classes,
                              field_classes=[],
                              data_type='page')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    #print(dataset.imgs)
    #print(dataset.imgData[0])
    #print(next(iter(dataloader)))
    #sys.exit(1)

    mrcnn = mask_rcnn.MaskRCNN()
    #print(mrcnn)

    mrcnn.to(device)

    """
    from coco_utils import get_coco_api_from_dataset
    from coco_eval import CocoEvaluator
    coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = _get_iou_types(mrcnn)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    print(coco_evaluator)
    mrcnn.load_state_dict(torch.load('mrcnn_model_resnet50_air.pth'))
    coco_evaluator, eval_values = mask_rcnn.evaluate(mrcnn, dataloader, device=device)
    print("eval_values: {}".format(eval_values))
    """


    # construct an optimizer
    params = [p for p in mrcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 20
    for epoch in range(num_epochs):
        mask_rcnn.train_one_epoch(mrcnn, optimizer, dataloader, device, epoch, print_freq=10)
        lr_scheduler.step()

    print(runlog)
    torch.save(mrcnn.state_dict(), "mrcnn_model_resnet50_air_page.pth")


    """
    import visualize
    mrcnn.load_state_dict(torch.load('mrcnn_model_car_resnet.pth'))
    mrcnn.eval()

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            # print(images)

            model_time = time.time()
            outputs = mrcnn(images)
            for _i, t in enumerate(outputs):
                print(images[_i])
                img = images[_i].cpu().clone()
                img = torchvision.transforms.ToPILImage()(img)
                # img.show()
                img = np.array(img)
                visualize.display_images([img])

                # img = images[_i].cpu().numpy()
                # img = img.transpose(1,2,0)
                print(t)
                boxes = t['boxes'].cpu().clone().numpy()
                labels = t['labels'].cpu().clone().numpy()
                scores = t['scores'].cpu().clone().numpy()
                masks = t['masks'].cpu().clone().numpy().transpose(1,2,3,0)[0]
                class_names = ['BG', 'C01', 'C02', 'C03', 'C04']
                print(boxes.shape, labels.shape, scores.shape, masks.shape)
                #print(np.where(masks[:, :, 0]))
                visualize.display_instances(img, boxes, masks, labels, class_names)
                sys.exit(1)
    """
    """
    import visualize
    img, target = dataset[0]
    #print(img)
    img = torchvision.transforms.ToPILImage()(img)
    img = np.array(img)
    visualize.display_images([img])

    #boxes, masks, class_ids, class_names
    boxes = np.array(target['boxes'])
    masks = np.array(target['masks']).transpose((1, 2, 0))
    class_ids = np.array(target['labels'])
    #class_names = dataset.page_classes
    class_names = dataset.field_classes
    print(type(boxes), type(masks), masks.shape)
    visualize.display_instances(img, boxes, masks, class_ids, class_names)
    """
