import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt

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

def features_exam(xdata, ydata, n_components=2, classes_name=[]):
    from sklearn.decomposition import PCA

    _, ax = plt.subplots(1, figsize=(5, 5))
    # Generate random colors
    colors = random_colors(len(classes_name))
    ax.axis('on')

    pca = PCA(n_components=2)
    newData = pca.fit_transform(xdata)
    features = {}
    for _f in range(len(classes_name)):
        features[_f] = []
    for _i, _d in enumerate(newData):
        features[ydata[_i]].append(_d)
    for _f in features.keys():
        features[_f] = np.array(features[_f])

    for _i, name in enumerate(classes_name):
        ax.plot(features[_i][:, 0], features[_i][:, 1],
                label=name, color=colors[_i], marker='+',
                linestyle='None')

    ax.legend(loc="upper right")
    plt.show()

def imshow(img,
           title=None,
           figsize=(4, 4),
           ax=None,
           is_display=False,
           is_save=[False, ""]):
    """Imshow for Tensor."""
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    if img[0] == 'npy':
        inp = img[1]
    elif img[0] == 'torch':
        print(type(img[1]))
        inp = img[1].byte()
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        #inp = std * inp + mean
        #inp = np.clip(inp, 0, 1)

    # Show area outside image boundaries.
    height, width = inp.shape[:2]
    print(height, width)
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('on')
    if title is not None:
        ax.set_title(title)

    if is_save[0]:
        print("save plot {}".format(is_save[1]))
        ax.imshow(inp)
        plt.savefig(is_save[1])

    if is_display:
        ax.imshow(inp)
        plt.show()

    plt.close('all')