import os.path as osp
import lmdb
import pickle
import time
import torch
import io

from PIL import Image
from torchvision.io import decode_jpeg, ImageReadMode

#
#  Code by Benjamin Vandersmissen. Contact me on Teams if there are any issues.
#  Benchmarks : 
#

class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None, backend='Vision'):
        """
            Initialize the Dataset object.
            
            Keyword Arguments:
                db_path : the location of the '.lmdb' object as a path-like (f.e. str, os.path).
                transform : the transformation done on each image, either None or a Callable that has a Torch.Tensor as input and as output.
                target_transform : the transformation done on each target / label, either None or a Callable that has a Torch.Tensor as input and as output.
                backend : either 'Pillow' or 'Vision'. The Pillow backend is also used by the ImageFolder class (https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html),
                                                       and as such the validation accuracy of the pretrained model is exactly the same as with the ImageFolder class. 
                                                       Vision is the TorchVision implementation which uses the nvjpeg library, which outputs a Tensor and can also be GPU-accelerated. 
                                                       However, the resulting JPEGs are slightly different from those decoded by Pillow. 
                                                       This results in a slightly lower accuracy on a pre-trained ResNet-50 model, (80.846% vs 79.826%), however we 
                                                       hypothesise that training the model from scratch with images decoded by nvjpeg will reach similar accuracies. 
                                                       !!! Right now, we recommend using Pillow as backend, because it is only slightly slower, but performs better with pretrained models. !!!
        """
        self.db_path = db_path
        self.environment = lmdb.open(db_path, subdir=osp.isdir(db_path),
                                readonly=True, lock=False,
                                readahead=False, meminit=False, 
                                metasync=False, sync=False)
        with self.environment.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))

        self.transform = transform
        self.target_transform = target_transform
        self.backend = backend

    def __getitem__(self, index):
        """
            Implements the map-style iteration of the Dataset, as specified in https://pytorch.org/docs/stable/data.html#map-style-datasets.
            
            Returns a (image, label) tuple of two torch.Tensors
        """        
        if index >= len(self): raise IndexError  # This will allow us to use iter(dataset), as well as some additional safety against user actions.
                
            
        with self.environment.begin(write=False) as txn:
            byteflow = txn.get(u'{}'.format(index).encode('ascii'))
        unpacked = pickle.loads(byteflow)        
        
        if self.backend == 'Vision':      
            imgbuf = bytearray(unpacked[0])  # Convert immutable (non-writeable) bytes object to mutable (writeable) bytearray object.
            imgbuf = torch.frombuffer(imgbuf, dtype=torch.uint8)

            img = decode_jpeg(imgbuf, mode=ImageReadMode.RGB, device='cpu')  # Currently only on CPU, because GPU decoding is in beta and requires CUDA >= 11.6: https://pytorch.org/vision/stable/generated/torchvision.io.decode_jpeg.html
            img = img.float() / 255

        elif self.backend == 'Pillow':
            img = Image.open(io.BytesIO(unpacked[0])).convert("RGB")
        
        else:
            raise ValueError(f"Backend {self.backend} is invalid. Either use 'Pillow' or 'Vision', the recommended option is 'Pillow'.")
        
        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
                
        return img, target

    def __len__(self):
        """
            Implements the map-style iteration of the Dataset, as specified in https://pytorch.org/docs/stable/data.html#map-style-datasets.
        """
        return self.length

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.db_path}) : containing {len(self)} samples.'

def imagenet(root='data', train=True, transform=None):
    if train:
        return LMDBDataset(db_path=root+'train.lmdb', transform=transform)
    else:
        return LMDBDataset(db_path=root+'val.lmdb', transform=transform)