import os
import sys
import cv2

import logging


import imgaug.augmenters as iaa

SEQ_AUG = iaa.SomeOf((1, 4), [
    iaa.MaxPooling([2, 4]),
    iaa.Affine(scale=(0.8, 1.2)),
    iaa.Affine(rotate=(-20, 20)),
    iaa.ScaleX((0.5, 1.5)),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
    iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    iaa.Fliplr(0.5),
    iaa.AddToHue((-50, 50)),
    iaa.Grayscale(alpha=(0.0, 1.0)),
])

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

def img_augmentation(img, augmentation):
    # Make augmenters deterministic to apply similarly to images and masks
    det = augmentation.to_deterministic()
    img = det.augment_image(img)
    return img, det


if __name__ == '__main__':
    root = "./db"
    save2dir = 'aug'
    aug_times = 2

    AUGDIR = path_join(nt_path(root), save2dir)
    if not os.path.isdir(AUGDIR):
        try:
            os.makedirs(AUGDIR)
        except:
            logging.error("cannot create folder at path:{}".format(AUGDIR))

    for rs, ds, fs in os.walk(root):
        if os.name == 'nt':
            folder = nt_path(rs)
        else:
            folder = rs
        if "tmp" in folder:
            continue
        for f in fs:
            if "json" in f:
                continue

            img = cv2.imread(f)
            img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(f).split('.jpg')[0]

            for t in range(aug_times):
                img_aug, det = img_augmentation(img_org, SEQ_AUG)
                save_name = basename+str(t)+'.jpg'
                cv2.imwrite(os.path.join(AUGDIR, save_name), img_aug)