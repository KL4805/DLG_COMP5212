import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms



def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)

    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.uniform_(-0.5, 0.5)
        
class LeNet(nn.Module):
    def __init__(self, input_dim, out_dim, act = 'relu'):
        super(LeNet, self).__init__()
        if act == 'sigmoid':
            self.act = nn.Sigmoid
        elif act == 'tanh':
            self.act = nn.Tanh
        elif act == 'relu':
            self.act = nn.ReLU
        self.body = nn.Sequential(
            nn.Conv2d(input_dim, 12, kernel_size=5, padding=5//2, stride=2),
            self.act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            self.act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            self.act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, out_dim)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act = 'relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out)
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, act='relu'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        if act == 'relu':
            self.act = nn.RelU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        # out = torch.sigmoid(out)
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels = 3, act = 'relu'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act=act)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1, act=act)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1, act=act)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1, act=act)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, act):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.act(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = F.adaptive_avg_pool2d(out, 1)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        return out


def ResNet18(nclass = 10, in_channels = 3, act = 'relu'):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=nclass, in_channels=in_channels, act=act)

def ResNet34(nclass = 10, in_channels = 3, act = 'relu'):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=nclass, in_channels=in_channels, act=act)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def alexnet32x32(in_channels=3, act=nn.ReLU(inplace=True)):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 5, 1, 2),
        act,
        nn.MaxPool2d(2, 2),  # 32 -> 16
        nn.Conv2d(64, 192, 5, 1, 2),
        act,
        nn.MaxPool2d(2, 2),  # 16 -> 8
        nn.Conv2d(192, 384, 3, 1, 1),
        act,
        nn.Conv2d(384, 256, 3, 1, 1),
        act,
        nn.Conv2d(256, 256, 3, 1, 1),
        act,
        nn.MaxPool2d(2, 2),  # 8 -> 4
        nn.AdaptiveAvgPool2d((4, 4))
        # adaptiveavgpool(a, b): for all input sized [B, C, H, W], we pool it to (B, C, a, b)
    )


class AlexNet(nn.Module):
    def __init__(self, nclass, act='sigmoid', in_channels=3):
        super(AlexNet, self).__init__()

        if act == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act == 'tanh':
            act_layer = nn.Tanh()
        else:
            act_layer = nn.Sigmoid()

        self.actname = act

        self.features = alexnet32x32(in_channels, act_layer)

        self.avgpool1x1 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 256),
            act_layer
        )
        outsize = 256

        self.cefc = nn.Linear(outsize, nclass)


        self.fc_layers = nn.ModuleList([self.fc, self.cefc])

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.actname == 'sigmoid':
                    nn.init.uniform_(m.weight, -0.3, 0.3)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity=self.actname)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        u = self.cefc(x)
        return u 
