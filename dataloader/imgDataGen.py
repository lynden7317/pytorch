import os
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
import torchvision

#from PIL import Image
import cv2
from imutils import paths

import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Imshow for Tensor."""
    print(type(inp))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.pause(1.0)  # pause a bit so that plots are updated
    plt.show()

class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.
        Arguments:
           data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source):
        print("nb_samples: {}".format(data_source))
        self.data_source = data_source
    def __iter__(self):
        return iter(range(len(self.data_source)))
    def __len__(self):
        return len(self.data_source)

class RoundRobinSampler(Sampler):
    """Samples elemnts with a,b,c,a,b,c,...
    """
    def __init__(self, data_source):
        self.listIDs = []
        self.partitionDict = {}
        self.data_source = data_source
        self.maxItem = 0

        for _i, _v in enumerate(self.data_source):
            _lab = os.path.basename(_v).split('_')[0]
            if _lab in self.partitionDict.keys():
                self.partitionDict[_lab][0].append(_i)
            else:
                self.partitionDict[_lab] = [[_i], 0]
        for _k in self.partitionDict.keys():
            if len(self.partitionDict[_k][0]) > self.maxItem:
                self.maxItem = len(self.partitionDict[_k][0])

        print("maxItem: {}".format(self.maxItem))
        for _i in range(self.maxItem):
            for _k in self.partitionDict.keys():
                _idx = self.partitionDict[_k][1] % len(self.partitionDict[_k][0])
                self.listIDs.append(self.partitionDict[_k][0][_idx])
                self.partitionDict[_k][1] += 1
        print("listIDs: {}".format(self.listIDs))

    def __iter__(self):
        return iter(self.listIDs)

    def __len__(self):
        return len(self.listIDs)

def ImageFolderDataLoader(data_dir, batch_size=4, shuffle=True, partition=['train', 'val']):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in partition}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=4)
                   for x in partition}

    dataset_sizes = {x: len(image_datasets[x]) for x in partition}
    if 'train' in partition:
        class_names = image_datasets['train'].classes
    else:
        class_names = []

    return dataloaders, dataset_sizes, class_names

class MyCustomDataset(Dataset):
    def __init__(self, args, transforms=None, height=224, width=224):
        # stuff
        self.datafolder = args["datafolder"]
        self.height = height
        self.width = width
        self.transforms = transforms

        self.datalist = []
        self.datalist = list(paths.list_images(self.datafolder))

    def __getitem__(self, index):
        # stuff
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        imgpath = self.datalist[index]
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = Image.open(imgpath).convert('RGB')
        #print(type(img))
        img_name = self.__img_name(imgpath)
        label = img_name.split("_")[0]

        img = torchvision.transforms.ToPILImage()(img).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        img = torchvision.transforms.Resize((self.height, self.width))(img)
        img = torchvision.transforms.ToTensor()(img)

        return (img, label)

    def __len__(self):
        # of how many examples(images?) you have
        return len(self.datalist)

    def __img_name(self, img_path):
        if os.name == 'nt':
            img_name = img_path.split("\\")[-1]
        else:
            img_name = img_path.split(os.path.sep)[-1]
        return img_name

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    args = {"datafolder": "./data/dataset"}

    # Define transforms (1)
    transforms = torchvision.transforms.Compose([torchvision.transforms.Pad(padding=32)])
    # Call the dataset
    custom_dataset = MyCustomDataset(args, transforms=transforms, height=256, width=256)

    sampler = RoundRobinSampler(custom_dataset.datalist) #SequentialSampler(custom_dataset.datalist)

    #dataloader = DataLoader(custom_dataset, batch_size=10, shuffle=True)
    dataloader = DataLoader(custom_dataset, batch_size=10, sampler=sampler)

    dataloaders, dataset_sizes, class_names = ImageFolderDataLoader("./data/hymenoptera_data", batch_size=10)
    print(dataset_sizes, class_names)
    #inputs, classes = next(iter(dataloader))
    #print(next(iter(dataloader)))

    for inputs, labels in dataloaders['train']:
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out)