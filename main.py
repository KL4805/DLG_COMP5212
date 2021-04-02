# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
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
                    help="Network architecture to test deep leakage. Supports LeNet, ResNet18, ResNet34, AlexNet")
parser.add_argument("--data", type=str, default="MNIST", 
                    help='Dataset to use. Supports MNIST, FashionMNIST, CIFAR10')
parser.add_argument("--iters", type = int, default = 300, 
                    help='Number of iterations to run DLG. Should be multiples of 10. ')
# parser.add_argument('--optim', type = str, default = 'lbfgs', 
#                     help='Optimizer to use. According to paper, lbfgs.')
parser.add_argument('--alg', type = str, default = 'DLG', 
                    help='Method to use. Supports DLG and iDLG. iDLG infers label first.')
parser.add_argument('--act', type = str, default = 'relu', 
                    help='Activation function to use. Supports tanh, sigmoid and relu.')
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

nclass = 10 if args.data in ['MNIST', 'FashionMNIST', 'CIFAR10'] else 10
nchannel = 1 if args.data in ['MNIST', 'FashionMNIST']  else 3
tp = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])
tt = transforms.ToPILImage()

img_index = args.index
# the target for deep leakage
gt_data = tp(dst[img_index][0]).to(device)
# Data range is from 0 to 1. 



gt_data = gt_data.view(1, *gt_data.size())
print(gt_data.shape)
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
print("Ground Truth Label is %d" % gt_label.item())
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label, num_classes = 10)

plt.imshow(tt(gt_data[0].cpu()))

from models.vision import LeNet, weights_init, ResNet18, ResNet34, AlexNet
if args.arch == 'LeNet5':
    net = LeNet(nchannel, nclass, args.act).to(device)
    net.apply(weights_init)
elif args.arch == 'ResNet18':
    net = ResNet18(nclass, nchannel, args.act).to(device)
    net.apply(weights_init)
elif args.arch == 'ResNet34':
    net = ResNet34(nclass, nchannel, args.act).to(device)
    net.apply(weights_init)
elif args.arch == 'AlexNet':
    net = AlexNet(nclass = nclass, in_channels=nchannel, act=args.act).to(device)
else:
    raise NotImplementedError("Only supports LeNet5, AlexNet, ResNet18 and ResNet34.")
print(net)

torch.manual_seed(1234)

criterion = cross_entropy_for_onehot

# compute original gradient, i.e. the gradient of the original data, label pair w.r.t minimizing the loss
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

# this would serve as a "label" for optimization
original_dy_dx = list((_.detach().clone() for _ in dy_dx))
print(original_dy_dx[-2].sum(-1))

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))
# if args.optim == 'lbfgs':
#     optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
# elif args.optim == 'adam':
#     optimizer = torch.optim.Adam([dummy_data, dummy_label], lr = 0.1)
if args.alg == 'DLG':
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
elif args.alg == 'iDLG':
    optimizer = torch.optim.LBFGS([dummy_data])
    infer_label = torch.argmin(torch.sum(original_dy_dx[-2], dim = -1), dim = -1).detach().reshape((1, )).requires_grad_(False)
    print("Inferred label by iDLG is %d" % infer_label.item())

history = []
start_time = time.time()
for iters in range(args.iters):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        if args.alg == 'DLG':
            dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        elif args.alg == 'iDLG':
            dummy_loss = F.cross_entropy(dummy_pred, infer_label)
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

if args.alg == 'DLG':
    print("Dummy label inferred by DLG is %d" % torch.argmax(dummy_label, dim = -1))
mse = (dummy_data - gt_data).pow(2).mean()
print("MSE between inferred and ground truth is %f" % mse)
print("PSNR between inferred and ground truth is %f" % (10 * torch.log10(1/mse)))
# dummy_npy = dummy_data[0].detach().cpu().numpy()
# gt_npy = gt_data[0].detach().cpu().numpy()
# print(dummy_npy.shape)
# print(gt_npy.shape)
# print("SSIM between inferred and ground truth is %f" % (structural_similarity(dummy_npy, gt_npy, gaussian_weights=True, sigma=1.5, use_sample_covariance = False)))
plt.figure(figsize=(12, 8))
for i in range(args.iters // 10):
    plt.subplot(args.iters // 100 + 1, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
