import torchvision
from functools import partial

from . import resnet, convnet, vgg, mobilenetv2
from .prunable_module import PrunableModule

all_models = ['WideResNet18', 'qWideResNet18', 'ResNet18', 'ResNet34', 'ResNet50', 'VGG11', 'VGG16', 'MobileNetv2', 'ConvNet']

def get_model(name, num_classes, tiny=False):
    if name not in all_models:
        raise Exception
    if tiny:
        if name == 'ResNet18':
            return partial(resnet.tiny_resnet18, num_classes=num_classes)
        elif name == 'ResNet34':
            return partial(resnet.tiny_resnet34, num_classes=num_classes)
        elif name == 'ResNet50':
            return partial(resnet.tiny_resnet50, num_classes=num_classes)
        elif name == 'WideResNet18':
            return partial(resnet.tiny_resnet18, num_classes=num_classes, width_mul=2)
        elif name == 'qWideResNet18':
            return partial(resnet.tiny_resnet18, num_classes=num_classes, width_mul=4)
        elif name == 'ConvNet':
            return partial(convnet.ConvNet, num_classes=num_classes)
        elif name == 'VGG11':
            return partial(vgg.vgg11, num_classes=num_classes)
        elif name == 'VGG16':
            return partial(vgg.vgg16_bn, num_classes=num_classes)
        else:
            raise Exception(f"{name} is not compatible with this input size")
    else:
        if name == 'ResNet18':
            return partial(resnet.resnet18, num_classes=num_classes)
        elif name == 'ResNet34':
            return partial(resnet.resnet34, num_classes=num_classes)
        elif name == 'ResNet50':
            return partial(resnet.resnet50, num_classes=num_classes)
        else:
            raise Exception(f"{name} is not compatible with this input size")
    