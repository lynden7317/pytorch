import os
import sys
import logging
import numpy as np
import cv2
from PIL import Image, ImageTk

from distutils.version import LooseVersion
import skimage.transform

import gvars

logging.basicConfig(level=logging.DEBUG)  # logging.DEBUG

def resize_image(img, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    # Keep track of image dtype and return results in the same dtype
    image_dtype = img.dtype
    logging.debug("<resize_image> dtype:{}".format(image_dtype))
    # Default window (x1, y1, x2, y2) and default scale == 1.
    h, w = img.shape[:2]
    if w <= 0:
        print('Error:resize_image:w<=0')
        raise ValueError('Error:resize_image:w<=0')

    if h <= 0:
        print('Error:resize_image:h<=0')
        raise ValueError('Error:resize_image:h<=0')

    window = (0, 0, w, h)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]

    if mode == "none":
        return img, window, scale, padding

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        img = resize(img, (round(h * scale), round(w * scale)), preserve_range=True)
        # image = skimage.transform.resize(image, (round(h * scale), round(w * scale)),
        #        order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = img.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        img = np.pad(img, padding, mode='constant', constant_values=0)
        window = (left_pad, top_pad, w + left_pad, h + top_pad)
    else:
        raise Exception("Mode {} not supported".format(mode))

    return img.astype(image_dtype), window, scale, padding


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
            preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

def nt_path(root):
    nts = [p for p in root.split('\\')]
    path = nts[0]
    for p in nts[1:]:
        path = path + '/' + p
    return path

class uiController(object):
    def __init__(self):
        super(uiController, self).__init__()

        self.openDir = None

    def delegator(self, *args, **kwargs):
        if 'name' in kwargs.keys():
            resultCB = {
                'fileIO':          lambda: self.CBfileIO(*args, **kwargs),
                'openDir':         lambda: self.CBopenDir(*args, **kwargs),
                'N1_images':       lambda:self.CBN1Images(*args, **kwargs),
            }[kwargs['name']]
        return resultCB

    def CBfileIO(self, *args, **kwargs):
        if kwargs['ftype'] == 'save':
            with open(kwargs['fpath'], 'w') as fid:
                for item in gvars.F_MARK:
                    fid.write("%s\n" % item)
        else:
            pass

    def CBopenDir(self, *args, **kwargs):
        print('open folder: ', kwargs)
        gvars.DIRPATH = kwargs['dirpath']
        gvars.F_NEXT = self._trace(kwargs['dirpath'])
        gvars.F_PRE = [[], -1, -1]
        gvars.F_MARK = []
        print('dirpath, f_next = {}, {}'.format(gvars.DIRPATH, gvars.F_NEXT))

    def CBN1Images(self, *args, **kwargs):
        print(args, kwargs)
        try:
            if len(gvars.F_PRE) > 0:
                pass
        except:
            print("Select the folder")
            return
        
        if kwargs['function'] == 'cont':
            print('cont')
            try:
                _tr = next(gvars.F_NEXT)
                while len(_tr) <= 0:
                    _tr = next(gvars.F_NEXT)
                print("cont: ", _tr)
                while True:
                    print("***: ", _tr)
                    basename = os.path.basename(_tr).split('.jpg')[0]
                    dirpath = os.path.dirname(_tr)
                    labpath = os.path.join(dirpath, basename + ".txt")
                    print(labpath)
                    if os.path.isfile(labpath):
                        gvars.F_PRE[0].append(_tr)
                        gvars.F_PRE[1] += 1
                        gvars.F_PRE[2] += 1
                        _tr = next(gvars.F_NEXT)
                        while len(_tr) <= 0:
                            _tr = next(gvars.F_NEXT)
                    else:
                        gvars.F_PRE[0].append(_tr)
                        gvars.F_PRE[1] += 1
                        gvars.F_PRE[2] += 1
                        break

                print(_tr)
                ind = gvars.F_PRE[1] - 1
                fname = gvars.F_PRE[0][ind]
                imgtk = self._readImg(fname)
                kwargs['panel'].imgtk = imgtk
                kwargs['panel'].config(image=imgtk)
                kwargs['info'].config(text=fname)
                kwargs['lab_total'].config(text=str(gvars.F_PRE[2]+1))
                gvars.F_PRE[1] = ind

                # reload label file
                basename = os.path.basename(fname).split('.jpg')[0]
                dirpath = os.path.dirname(fname)
                labpath = os.path.join(dirpath, basename + ".txt")
                try:
                    with open(labpath, 'r') as fid:
                        li = fid.readline()
                        rads = li.split(" ")
                        kwargs['rad1Var'].set(int(rads[0]))
                        kwargs['rad2Var'].set(int(rads[1]))
                        kwargs['rad3Var'].set(int(rads[2]))
                except:
                    print("no label file")


            except StopIteration:
                kwargs['info'].config(text='All Done')
            except:
                print("except return")
                return None


        elif kwargs['function'] == 'prev':
            print('prev')
            if gvars.F_PRE[2] > 0 and gvars.F_PRE[1] > 0:
                ind = gvars.F_PRE[1] - 1
                fname = gvars.F_PRE[0][ind]
                imgtk = self._readImg(fname)
                kwargs['panel'].imgtk = imgtk
                kwargs['panel'].config(image=imgtk)
                kwargs['info'].config(text=fname)
                gvars.F_PRE[1] = ind
                #print('PRE: {}'.format(gvars.F_PRE))

                # reload label file
                basename = os.path.basename(fname).split('.jpg')[0]
                dirpath = os.path.dirname(fname)
                labpath = os.path.join(dirpath, basename + ".txt")
                try:
                    with open(labpath, 'r') as fid:
                        li = fid.readline()
                        rads = li.split(" ")
                        kwargs['rad1Var'].set(int(rads[0]))
                        kwargs['rad2Var'].set(int(rads[1]))
                        kwargs['rad3Var'].set(int(rads[2]))
                except:
                    print("no label file")
            else:
                return None
        elif kwargs['function'] == 'next':
            print('next')
            if gvars.F_PRE[1] == gvars.F_PRE[2]:
                try:
                    _tr = next(gvars.F_NEXT)
                    while len(_tr) <= 0:
                        _tr = next(gvars.F_NEXT)

                    gvars.F_PRE[0].append(_tr)
                    gvars.F_PRE[1] += 1
                    gvars.F_PRE[2] += 1
                    #print('*next PRE: {}'.format(gvars.F_PRE))
                    fname = _tr
                    imgtk = self._readImg(_tr)
                    kwargs['panel'].imgtk = imgtk
                    kwargs['panel'].config(image=imgtk)
                    kwargs['info'].config(text=_tr)
                    kwargs['lab_total'].config(text=str(gvars.F_PRE[2]+1))
                except StopIteration:
                    kwargs['info'].config(text='Done')
                except:
                    return None
            else:
                ind = gvars.F_PRE[1] + 1
                fname = gvars.F_PRE[0][ind]
                imgtk = self._readImg(fname)
                kwargs['panel'].imgtk = imgtk
                kwargs['panel'].config(image=imgtk)
                kwargs['info'].config(text=fname)
                gvars.F_PRE[1] = ind
                #print('**next PRE: {}'.format(gvars.F_PRE))

            # create mark file
            if len(gvars.F_PRE[0]) > 1:
                ind = gvars.F_PRE[1] - 1
                pre_fname = gvars.F_PRE[0][ind]
                basename = os.path.basename(pre_fname).split('.jpg')[0]
                dirpath = os.path.dirname(pre_fname)
                print("save labfile", pre_fname, basename, dirpath)
                print(kwargs['rad1Var'].get(), kwargs['rad2Var'].get(), kwargs['rad3Var'].get())
                labpath = os.path.join(dirpath, basename+".txt")
                
                with open(labpath, 'w') as fid:
                    labstr = str(kwargs['rad1Var'].get())+" "+str(kwargs['rad2Var'].get())+" "+str(kwargs['rad3Var'].get())
                    fid.write(labstr)
            
            # load mark file, if exist
            basename = os.path.basename(fname).split('.jpg')[0]
            dirpath = os.path.dirname(fname)
            labpath = os.path.join(dirpath, basename+".txt")
            if os.path.isfile(labpath):
                #print("labpath: ", labpath)
                try:
                    with open(labpath, 'r') as fid:
                        li = fid.readline()
                        rads = li.split(" ")
                        kwargs['rad1Var'].set(int(rads[0]))
                        kwargs['rad2Var'].set(int(rads[1]))
                        kwargs['rad3Var'].set(int(rads[2]))
                except:
                    pass
            else:
                kwargs['setRads2Default']()
        
        elif kwargs['function'] == 'mark':
            print('mark')
            try:
                mark_path = kwargs['info'].cget('text')
                if mark_path in gvars.F_MARK:
                    pass
                else:
                    gvars.F_MARK.append(mark_path)
                kwargs['lab_mark'].config(text=len(gvars.F_MARK))
            except:
                return None
        else:
            return None

    def _readImg(self, path):
        img = cv2.imread(path)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        w, h = img.shape[1], img.shape[0]
        #print('w: {}, h:{}'.format(w, h))
        img_re, _, _, _ = resize_image(cv2image, min_dim=512, max_dim=512)
        current_image = Image.fromarray(img_re)
        #print('resize: {}'.format(img_re.shape))
        imgtk = ImageTk.PhotoImage(image=current_image)
        return imgtk

    def _trace(self, folder):
        for root, dirs, files in os.walk(folder):
            jpgs = []
            for f in files:
                #print(f)
                if f.split('.')[1] in ['jpg']:
                    fname = f.split('.')[0]
                    if os.name == 'nt':
                        _path = nt_path(os.path.join(root, f))
                    else:
                        _path = os.path.join(root, f)
                    jpgs.append(_path)
                    
                    #if len(fname.split('_')) > 1:
                    #    jpgs.append(os.path.join(root, f))
            #print(jpgs)
            for j in jpgs:
                yield j
            #yield jpgs