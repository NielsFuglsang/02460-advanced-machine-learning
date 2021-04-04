# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms

from utils import (label_to_onehot, cross_entropy_for_onehot, init_data, create_loss_measure)
from models.vision import LeNet, weights_init

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25", help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="", help='the path to customized image.')
parser.add_argument('--inittype', type=str, default="uniform", help='the data initialization type. (uniform/gaussian).')
parser.add_argument('--measure', type=str, default="euclidean", help='the distance measure. (euclidean/gaussian.')
parser.add_argument('--Q', type=str, default=1, help='set value of Q in gaussian measure.')
args = parser.parse_args()

torch.manual_seed(1234)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

def format_image(dst, index):
    """Format CIFAR image to tensor."""
    tp = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])

    gt_data = tp(dst[index][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    return gt_data

def format_label(dst, index):
    """Format CIFAR label to tensor"""
    gt_label = torch.Tensor([dst[index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)
    return gt_onehot_label

#
tt = transforms.ToPILImage()

dst = datasets.CIFAR100("~/.torch", download=True)
# Specify indices.
indices = [25, 26]

# Get ground truth batch of images and labels.
images = [format_image(dst, idx) for idx in indices]
labels = [format_label(dst, idx) for idx in indices]
gt_data = torch.cat(images, 0)
gt_onehot_label = torch.cat(labels, 0)


net = LeNet().to(device)

net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute original gradient
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label
dummy_data, dummy_label = init_data(gt_data, gt_onehot_label, device, inittype=args.inittype)


loss_measure = create_loss_measure(args, original_dy_dx)
optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.1)

history = []

# gt_im = gt_data[0].cpu().numpy().transpose((1, 2, 0))

n = 50
val_size = 5
for iters in range(n):

    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        grad_diff = loss_measure(original_dy_dx, dummy_dy_dx)

        grad_diff.backward()

        return grad_diff

    optimizer.step(closure)

    # Validation.
    if iters % val_size == 0:
        current_loss = closure()
        print(iters, "%.10f" % current_loss.item())

        history.append([tt(dummy_data[0].cpu()), tt(dummy_data[1].cpu())])

        dummy_im = dummy_data[0].cpu().detach().numpy().transpose((1, 2, 0))

fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0][i].imshow(history[i][0])
    axes[1][i].imshow(history[i][1])
    axes[0][i].set_title(f"it={i * val_size}")
    axes[1][i].set_title(f"it={i * val_size}")
    axes[0][i].axis('off')
    axes[1][i].axis('off')
plt.show()

