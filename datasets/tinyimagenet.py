import torch
import os
import pathlib

file_loc = pathlib.Path(__file__).parent.resolve()

class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, transforms):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms
        assert len(self.imgs) == len(self.labels)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img, lbl = self.imgs[i], self.labels[i]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, lbl

def tinyimagenet(root='data', train=True, transform=None):
    data = torch.load(os.path.join(file_loc, root, 'tinyimagenet.pt'), map_location='cpu', weights_only=True)

    img_key = 'images_train' if train else 'images_val'
    lbl_key = 'labels_train' if train else 'labels_val'

    imgs, lbls = data[img_key], data[lbl_key]
    imgs = imgs / 255.0
    return TinyImageNet(imgs, lbls, transform)
        