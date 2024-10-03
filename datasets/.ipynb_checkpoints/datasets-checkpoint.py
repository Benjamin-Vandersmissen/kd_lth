from .cub200 import cub200
from .imagenet import imagenet
from .tinyimagenet import tinyimagenet

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from functools import partial

all_datasets = ['CIFAR10', 'CIFAR100', 'TinyImageNet', 'ImageNet', 'CUB200']

def get_dataset(dataset, train=True):
    if dataset not in all_datasets:
        raise Exception
    
    if dataset == 'CIFAR10':
        num_classes = 10
        eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_transform = transforms.Compose([eval_transform, transforms.RandomCrop([32, 32], padding=4), transforms.RandomHorizontalFlip(0.5)])
        ds_fn = partial(CIFAR10, download=True, root='data')
        
    elif dataset == 'CIFAR100':
        num_classes = 100
        eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        train_transform = transforms.Compose([eval_transform, transforms.RandomCrop([32, 32], padding=4), transforms.RandomHorizontalFlip(0.5)])
        ds_fn = partial(CIFAR100, download=True, root='data')

    elif dataset == 'CUB200':
        num_classes = 200 
        base_transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])
        train_transform = transforms.Compose([base_transform, transforms.RandomHorizontalFlip(p=0.5), transforms.RandomResizedCrop(224)])
        eval_transform = transforms.Compose([base_transform, transforms.Resize(224), transforms.CenterCrop(224)])
        ds_fn = partial(cub200, root='data')
        
    elif dataset =='TinyImageNet':
        num_classes = 200
        eval_transform = transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
        train_transform = transforms.Compose([eval_transform, transforms.RandomResizedCrop(64, scale=(0.1, 1.0), ratio=(0.8, 1.25)), transforms.RandomHorizontalFlip()])
        ds_fn = partial(tinyimagenet, root='data')
        
    elif dataset == 'ImageNet':
        num_classes = 1000
        base_transforms = transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
        train_transform = transforms.Compose([base_transforms, transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(0.8, 1.25)), transforms.RandomHorizontalFlip()])
        eval_transform = transforms.Compose([base_transforms, transforms.Resize(256), transforms.CenterCrop(224)])
        ds_fn = partial(imagenet, root='/datasets_antwerp/ImageNet/')

    if train:
        return ds_fn(transform=train_transform, train=train), num_classes
    else:
        return ds_fn(transform=eval_transform, train=train), num_classes