import cv2
import scipy
import random
import numpy as np
import skimage
import skimage.transform
import warnings
from distutils.version import LooseVersion


def img_augmentation(img, augmentation):
    # Make augmenters deterministic to apply similarly to images and masks
    det = augmentation.to_deterministic()
    img = det.augment_image(img)
    return img, det

def mask_augmentation(img, mask, det):
    import imgaug
    MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                       "Fliplr", "Flipud", "CropAndPad",
                       "Affine", "PiecewiseAffine", "Multiply", "Grayscale", "AdditiveGaussianNoise"]

    def hook(images, augmenter, parents, default):
        """Determines which augmenters to apply to masks."""
        return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

    # Store shapes before augmentation to compare
    img_shape = img.shape
    mask_shape = mask.shape

    mask = det.augment_image(mask, hooks=imgaug.HooksImages(activator=hook))
    # Verify that shapes didn't change
    assert img.shape == img_shape, "Augmentation shouldn't change image size"
    assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

    return mask


def gray_3_ch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _timg = np.zeros((gray.shape[0], gray.shape[1], 3))
    _timg[:, :, 0] = gray
    _timg[:, :, 1] = gray
    _timg[:, :, 2] = gray
    return _timg

def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (x1, y1, x2, y2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (x2, y2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (x1, y1, x2, y2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([w - 1, h - 1, w - 1, h - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)

def denorm_mask(mask, scale, padding, outshape):
    print(padding)
    # remove padding
    x_pad = padding[0][0]+padding[0][1]
    y_pad = padding[1][0]+padding[1][1]
    unpad_mask = np.zeros((mask.shape[0]-x_pad, mask.shape[1]-y_pad))
    unpad_mask = mask[0+padding[0][0]:mask.shape[0]-padding[0][1], 0+padding[1][0]:mask.shape[1]-padding[1][1]]
    print(unpad_mask.shape)
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        de_mask = scipy.ndimage.zoom(unpad_mask, zoom=[1/scale, 1/scale], order=0)
    # confirm to the real output size
    out = np.zeros((outshape[0], outshape[1]))
    out[0:de_mask.shape[0], 0:de_mask.shape[1]] = de_mask
    print(out.shape)
    return out

def moldImage_resnet(img):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image colors in RGB order.
    """
    #print('moldImage_resnet')
    img[:,:,:] = img[:,:,:].astype('float32')
    img[:,:,:] = img[:,:,:] - np.array([123.7, 116.8, 103.9]) #RGB
    return img


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (x1, y1, x2, y2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (x1, y1, x2, y2) and default scale == 1.
    h, w = image.shape[:2]
    if w <= 0:
        print('Error:resize_image:w<=0')
        raise ValueError('Error:resize_image:w<=0')

    if h <= 0:
        print('Error:resize_image:h<=0')
        raise ValueError('Error:resize_image:h<=0')

    window = (0, 0, w, h)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

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
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)
        #image = skimage.transform.resize(image, (round(h * scale), round(w * scale)),
        #        order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (left_pad, top_pad, w + left_pad, h + top_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        #window = (top_pad, left_pad, h + top_pad, w + left_pad)
        window = (left_pad, top_pad, w + left_pad, h + top_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        #crop = (y, x, min_dim, min_dim)
        crop = (x, y, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        #y, x, h, w = crop
        x, y, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
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