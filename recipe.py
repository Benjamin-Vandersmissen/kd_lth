import torch
from torchvision import transforms
from functools import partial
import os

from models import get_model
from datasets import get_dataset
from utils import Step


class TrainingRecipe(object):
    arguments = {'train': ('The amount of total training episodes. This number includes eventual pretraining steps.', int),
                 'lr': ('The initial learning rate.', float),
                 'bs': ('Batch size', int),
                 'decay': ('The weight decay applied during training', float),
                 'momentum': ('The momentum of the SGD optimizer', float),
                 'warmup': ('Linear learning rate warmup for the SGD optimizer', float),
                 'lr_schedule': ('The LR schedule for the SGD optimizer. Formatted as following : *typ*,*comma-separated args*', str),
                 'pretrain_its': ('The amount of pretraining required for LTH', int)
                }
    
    def __init__(self, train, lr, bs, lr_schedule, weight_decay, momentum, warmup, pretrain_its=0):
        self.train = train
        self.lr = lr
        self.bs = bs
        self.lr_schedule = lr_schedule
        self.decay = weight_decay
        self.momentum = momentum
        self.warmup = warmup
        self.pretrain_its = pretrain_its
    
    def train_epochs(self):
        return self.train
    
    def learning_rate(self):
        return self.lr
    
    def batch_size(self):
        return self.bs
    
    def weight_decay(self):
        return self.decay
    
    def optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.decay) 
    
    def lr_scheduler(self, optimizer, cur_ep=None):
        typ, milestones = self.lr_schedule[0], self.lr_schedule[1:]
        
        if typ == 'step':
            # schedule_lambdas = [warmup_lambda]+[fn(milestone) for milestone in milestones]  
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: np.product([l(it) for l in schedule_lambdas]))
            pass
        elif typ == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=milestones[0])
        elif typ == 'none':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: x)

        if cur_ep is not None:
            for _ in range(cur_ep):
                scheduler.step()
        
        return scheduler
    
    @classmethod
    def add_to_parser(cls, parser):
        for argument, (desc, typ) in cls.arguments.items():
            parser.add_argument(f'--{argument}', required=False, type=typ, help=desc)
        return parser
    
    def override_from_args(self, args):
        for argument, (_, typ) in self.arguments.items():
            if hasattr(args, argument) and getattr(args, argument) is not None:
                setattr(self, argument, getattr(args, argument))
            if argument == 'lr_schedule':
                self.lr_schedule = self.lr_schedule.split(',')
                self.lr_schedule = [self.lr_schedule[0]] + [float(s) for s in self.lr_schedule[1:]]  # First value is the type of scheduler (e.g. cosine or linear), the next are arguments for the scheduler
        return self


def training_recipe(dataset, model, args):
    "Returns the following tuple : (train loader, test loader, model generator, recipe hyperparameters)"

    
    train, num_classes = get_dataset(dataset, train=True)
    val, _ = get_dataset(dataset, train=False)
    
    model_fn = get_model(model, num_classes, tiny = dataset in ['CIFAR10', 'CIFAR100', 'TinyImageNet'])
    
    if dataset in ['CIFAR10', 'CIFAR100'] and model == 'ConvNet':
        recipe = TrainingRecipe(train=200, lr=0.1, bs=256, weight_decay=5e-4, momentum=0.9, warmup=0, lr_schedule='cosine,200').override_from_args(args)
    elif dataset in ['CIFAR10', 'CIFAR100'] and model == 'VGG16':
        recipe = TrainingRecipe(train=200, lr=0.1, bs=256, weight_decay=5e-4, momentum=0.9, warmup=0, lr_schedule='cosine,200').override_from_args(args)
    elif dataset in ['CIFAR10', 'CIFAR100'] and model in ['ResNet18', 'WideResNet18', 'qWideResNet18', 'ResNet34', 'ResNet50']:
        recipe = TrainingRecipe(train=200, lr=0.1, bs=256, weight_decay=5e-4, momentum=0.9, warmup=0, lr_schedule='cosine,200', pretrain_its=1000).override_from_args(args)
    elif dataset in ['CIFAR10', 'CIFAR100'] and model == 'MobileNetv2':
        recipe = TrainingRecipe(train=200, lr=0.1, bs=256, weight_decay=5e-4, momentum=0.9, warmup=0, lr_schedule='cosine,200').override_from_args(args)
    elif dataset == 'CUB200' and model == 'VGG11':
        recipe = TrainingRecipe(train=200, lr=0.2, bs=256, weight_decay=5e-4, momentum=0.9, warmup=0, lr_schedule='cosine,200').override_from_args(args)
    elif dataset == 'TinyImageNet' and model in ['ResNet18', 'WideResNet18', 'qWideResNet18', 'ResNet34', 'ResNet50']:
        recipe = TrainingRecipe(train=200, lr=0.2, bs=256, weight_decay=1e-4, momentum=0.9, warmup=0, lr_schedule='cosine,200', pretrain_its=1000).override_from_args(args)
    elif dataset == 'TinyImageNet' and model in ['VGG16']:
        recipe = TrainingRecipe(train=200, lr=0.2, bs=256, weight_decay=1e-4, momentum=0.9, warmup=0, lr_schedule='cosine,200').override_from_args(args)
    elif dataset == 'ImageNet' and model == 'ResNet50':
        recipe = TrainingRecipe(train=90, lr=0.1, bs=256, weight_decay=1e-4, momentum=0.9, warmup=0, lr_schedule='cosine,90').override_from_args(args)
    else:
        raise Exception("Unknown model dataset combination!")

    trainloader = torch.utils.data.DataLoader(train, recipe.batch_size(), num_workers=len(os.sched_getaffinity(0)), shuffle=True)
    testloader = torch.utils.data.DataLoader(val, recipe.batch_size(), num_workers=len(os.sched_getaffinity(0)))
    return trainloader, testloader, model_fn, recipe, num_classes
        
        
        

    
