'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Source : https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    expansion = 1
    def __init__(self, fan_in, fan_out, downsample=False):
        super(Block, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(fan_in, fan_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(fan_out)
        self.conv2 = nn.Conv2d(fan_out, fan_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(fan_out)

        if downsample or fan_in != fan_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(fan_in, fan_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(fan_out)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        # Main path
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))

        # Add shortcut
        o += self.shortcut(x)
        return F.relu(o)


class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, fan_in, fan_out, downsample=False):
        super(BottleNeckBlock, self).__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(fan_in, fan_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(fan_out)
        self.conv2 = nn.Conv2d(fan_out, fan_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(fan_out)
        self.conv3 = nn.Conv2d(fan_out, self.expansion*fan_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*fan_out)

        if downsample or fan_in != self.expansion*fan_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(fan_in, self.expansion*fan_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*fan_out)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        # Main path
        o = F.relu(self.bn1(self.conv1(x)))
        o = F.relu(self.bn2(self.conv2(o)))
        o = self.bn3(self.conv3(o))

        # Add shortcut
        o += self.shortcut(x)
        return F.relu(o)


class ResNet(nn.Module):  # This is only suitable for 64x64 input or 32x32 input (i.e., CIFAR & Tiny-ImageNet)
    def __init__(self, blueprint, num_classes, blocktype: callable = Block, typ='cifar'):
        super(ResNet, self).__init__()

        self.typ = typ
        
        cur_w = blueprint[0][0]

        if typ == 'imagenet':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=cur_w, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif typ == 'cifar' or typ == 'tinyimagenet':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=cur_w, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cur_w)
        blocks = []
        for (width, downsample) in blueprint:
            blocks.append(blocktype(cur_w, width, downsample))
            cur_w = blocktype.expansion*width
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(cur_w, num_classes)

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        if self.typ == 'imagenet':
            o = self.maxpool1(o)
        o = self.blocks(o)
        o = F.avg_pool2d(o, o.shape[-1])
        o = o.view((o.shape[0], -1))
        return self.fc(o)


def tiny_resnet18(num_classes=10, width_mul=1):
    blueprint = [(width_mul*64, False), (width_mul*64, False),
             (width_mul*128, True), (width_mul*128, False),
             (width_mul*256, True), (width_mul*256, False),
             (width_mul*512, True), (width_mul*512, False)]

    return ResNet(blueprint, num_classes=num_classes)

def tiny_resnet20(num_classes=10):
    blueprint = [(16, False), (16, False), (16, False),
                 (32, True), (32, False), (32, False),
                 (64, True), (64, False), (64, False)]
    return ResNet(blueprint, num_classes=num_classes)

def tiny_resnet34(num_classes=10):
    blueprint = [(64, False), (64, False), (64, False),
                 (128, True), (128, False), (128, False), (128, False),
                 (256, True), (256, False), (256, False), (256, False), (256, False), (256, False),
                 (512, True), (512, False), (512, False), (512, False)]
    return ResNet(blueprint, num_classes=num_classes)

def tiny_resnet50(num_classes=10):
    blueprint = [(64, False), (64, False), (64, False),
                 (128, True), (128, False), (128, False), (128, False),
                 (256, True), (256, False), (256, False), (256, False), (256, False), (256, False),
                 (512, True), (512, False), (512, False), (512, False)]
    return ResNet(blueprint, blocktype=BottleNeckBlock, num_classes=num_classes)

def resnet18(num_classes=1000):
    blueprint = [(64, False), (64, False),
                 (128, True), (128, False),
                 (256, True), (256, False),
                 (512, True), (512, False)]

    return ResNet(blueprint, num_classes=num_classes, typ='imagenet')

def resnet34(num_classes=1000):
    blueprint = [(64, False), (64, False), (64, False),
                 (128, True), (128, False), (128, False), (128, False),
                 (256, True), (256, False), (256, False), (256, False), (256, False), (256, False),
                 (512, True), (512, False), (512, False), (512, False)]

    return ResNet(blueprint, num_classes=num_classes, typ='imagenet')

def resnet50(num_classes=1000):
    blueprint = [(64, False), (64, False), (64, False),
                 (128, True), (128, False), (128, False), (128, False),
                 (256, True), (256, False), (256, False), (256, False), (256, False), (256, False),
                 (512, True), (512, False), (512, False), (512, False)]
    return ResNet(blueprint, num_classes=num_classes, blocktype=BottleNeckBlock, typ='imagenet')
