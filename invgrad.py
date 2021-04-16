# This code implements the paper "Inverting Gradients-How easy is it to break privacy in federated learning"
# Note: This implementation borrows from https://github.com/JonasGeiping/invertinggradients
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
parser = argparse.ArgumentParser(description='Inverting Gradients-How easy is it to break privacy in federated learning')
parser.add_argument('--index', type=int, default=25,
                    help='the index for leaking images on the data.')
parser.add_argument("--arch", type=str, default='LeNet5', 
                    help="Network architecture to test deep leakage. Supports LeNet5, ResNet18, ResNet34, AlexNet")
parser.add_argument("--data", type=str, default="MNIST", 
                    help='Dataset to use. Supports MNIST, FashionMNIST, CIFAR10')
parser.add_argument("--iters", type = int, default = 500, 
                    help='Number of iterations to run DLG. Should be multiples of 10. ')
# parser.add_argument('--optim', type = str, default = 'adam', 
#                     help='Optimizer to use. According to paper, adam.')
# parser.add_argument('--alg', type = str, default = 'DLG', 
#                     help='Method to use. Supports DLG and iDLG. iDLG infers label first.')
parser.add_argument('--alpha', type = float, default = 0.01, 
                    help = 'The total variation penalty. According to the paper, it should have minor influence. ')
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

torch.manual_seed(1234)
criterion = cross_entropy_for_onehot

# compute original gradient, i.e. the gradient of the original data, label pair w.r.t minimizing the loss
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

# this would serve as a "label" for optimization
original_dy_dx = list((_.detach().clone() for _ in dy_dx))
print(original_dy_dx[-2].sum(-1))

dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

# by default, InvGrad uses iDLG, and used Adam Optimizer
optimizer = torch.optim.Adam([dummy_data], lr = 0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
    milestones=[args.iters // 2.667, args.iters // 1.6, args.iters // 1.142], gamma=0.1)   # 3/8 5/8 7/8, specified by the paper
infer_label = torch.argmin(torch.sum(original_dy_dx[-2], dim = -1), dim = -1).detach().reshape((1, )).requires_grad_(False)
print("Inferred label by iDLG is %d" % infer_label.item())

def total_variation(x):
    # computes the total varation of an image
    # x: [1, channel, lng, lat]
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    return dx + dy

history = []
start_time = time.time()
for iters in range(args.iters):
    dummy_data.data = torch.clamp(dummy_data.data, min = 0, max = 1)
    def closure():
        
        optimizer.zero_grad()
        net.zero_grad()
        dummy_pred = net(dummy_data)
        dummy_loss = F.cross_entropy(dummy_pred, infer_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        # compute cosine loss
        loss = 0
        dgnorm = 0
        gtnorm = 0
        for dg, gt in zip(dummy_dy_dx, original_dy_dx):
            loss -= (dg * gt).sum()
            dgnorm += dg.pow(2).sum()
            gtnorm += gt.pow(2).sum()

        loss = 1 + loss / (dgnorm.sqrt() * gtnorm.sqrt())
        loss += args.alpha * total_variation(dummy_data)
        # print(loss)
        loss.backward()
        # take sign of the loss
        dummy_data.grad.sign_()
        return loss
    optimizer.step(closure)
    scheduler.step()
    if iters % 10 == 0:
        rec_loss = closure()
        print("Time: %.2fs, Iter %d, Loss %.7f" % (time.time() - start_time, iters, rec_loss.item()))
        mse = (dummy_data - gt_data).pow(2).mean()
        print("MSE between inferred and ground truth is %f" % mse.item())
        history.append(tt(dummy_data[0].detach().cpu()))
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
