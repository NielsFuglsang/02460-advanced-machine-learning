# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint
import random

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

from utils import (
    label_to_onehot,
    cross_entropy_for_onehot,
    init_data,
    create_loss_measure,
    format_image,
    format_label,
    make_reconstruction_plots,
)
from models.vision import LeNet, ResNet, weights_init, resnet20, _weights_init

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25", help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="", help='the path to customized image.')
parser.add_argument('--inittype', type=str, default="uniform", help='the data initialization type. (uniform/gaussian).')
parser.add_argument('--measure', type=str, default="euclidean", help='the distance measure. (euclidean/gaussian.')
parser.add_argument('--Q', type=int, default=1, help='set value of Q in gaussian measure.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images to process in the reconstruction.')
parser.add_argument('--n_iters', type=int, default=300, help='Number iterations during the reconstruction.')
parser.add_argument('--val_size', type=int, default=50, help='Denotes the interval size of data collection between iterations.')
args = parser.parse_args()

random.seed(1337)
torch.manual_seed(1234)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)


batch_size = args.batch_size
n_iters = args.n_iters
val_size = args.val_size

tt = transforms.ToPILImage()

dst = datasets.CIFAR100("~/.torch", download=True)

# Specify indices.
indices = random.choices(list(range(len(dst))), k=batch_size)

# Get ground truth batch of images and labels.
images = [format_image(dst, idx, device) for idx in indices]
labels = [format_label(dst, idx, device) for idx in indices]
gt_data = torch.cat(images, 0)
gt_onehot_label = torch.cat(labels, 0)

#LeNet
net = LeNet().to(device)
net.apply(weights_init)

#ResNet20
#net = resnet20().to(device)
#net.apply(_weights_init)

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
for iters in range(n_iters):

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

        history.append([tt(dummy_data[i].cpu()) for i in range(batch_size)])


make_reconstruction_plots(
    history,
    batch_size,
    val_size,
    indices,
    dst, 
    figsize=(12, 8)
)

