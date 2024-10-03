import torch
import torchvision
import os
import PIL

def cub200(root='data',train=True, transform=None):
    root = os.path.join(root, 'CUB200-2011', 'CUB_200_2011')

    split = []
    with open(os.path.join(root, 'train_test_split.txt')) as f:
        for line in f.readlines():
            idx, is_train = map(int, line.split(' '))
            if is_train == train:
                split.append(idx)

    ds = torchvision.datasets.ImageFolder(os.path.join(root, 'images'), transform = transform)
    ds = torch.utils.data.Subset(ds, split)
    return ds
    
    # imgs = []
    # with open(os.path.join(root, 'images.txt')) as f:
    #     for line in f.readlines():
    #         idx, fname = line.split(' ')
    #         if int(idx) in split:
    #             imgs.append(PIL.Image.open(os.path.join(root, fname)))

    # lbls = []
    # with open(os.path.join(root, 'image_class_labels.txt')) as f:
    #     for line in f.readlines():
    #         idx, lbl = map(line.split(' '), int)
    #         if idx in split:
    #             lbls.append(lbl - 1) # Class labels start from 1, rather than 0 in the text 
