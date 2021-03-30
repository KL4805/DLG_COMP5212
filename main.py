# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import time
print("Pytorch version:", torch.__version__, "torchvision version:", torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default=25,
                    help='the index for leaking images on MNIST.')
parser.add_argument("--arch", type=str, default='LeNet5', 
                    help="Network architecture to test deep leakage. Supports LeNet, ResNet18, ResNet34")
parser.add_argument("--data", type=str, default="MNIST", 
                    help='Dataset to use. Supports MNIST, FashionMNIST, CIFAR10')
parser.add_argument("--iters", type = int, default = 300, 
                    help='Number of iterations to run DLG. Should be multiples of 10. ')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

if args.data == "MNIST":
    dst = datasets.MNIST("./", download = True)
elif args.data == 'FashionMNIST':
    dst = datasets.FashionMNIST('./', download = True)
elif args.data == 'CIFAR10':
    dst = datasets.CIFAR10("./CIFAR10", download = True)
else:
    raise NotImplementedError("Only supports MNIST, FashionMNIST, CIFAR10")
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
# the target for deep leakage
gt_data = tp(dst[img_index][0]).to(device)



gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes = 10)

plt.imshow(tt(gt_data[0].cpu()))

from models.vision import LeNet, weights_init, ResNet18, ResNet34
if args.arch == 'LeNet5':
    net = LeNet(1, 10).to(device)
elif args.arch == 'ResNet18':
    net = ResNet18()
elif args.arch == 'ResNet34':
    net = ResNet34()
else:
    raise NotImplementedError("Only supports LeNet5, ResNet18 and ResNet34.")

torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute original gradient, i.e. the gradient of the original data, label pair w.r.t minimizing the loss
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

# this would serve as a "label" for optimization
original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


history = []
start_time = time.time()
for iters in range(args.iters):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    if iters % 10 == 0: 
        current_loss = closure()
        print("Time: %.2fs, Iter %d, Loss %.4f" % (time.time() - start_time, iters, current_loss.item()))
        history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(args.iters // 10):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
