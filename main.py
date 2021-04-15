"""Old code."""


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

from models.vision import LeNet, weights_init
from utils import (
    label_to_onehot,
    cross_entropy_for_onehot,
    init_data,
    create_loss_measure,
)

print(torch.__version__, torchvision.__version__)


parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--data', type=str, default="CIFAR",
                    help='dataset to choose from in torchvision.datasets (CIFAR, MNIST etc.).')
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on dataset.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
parser.add_argument('--inittype', type=str, default="uniform",
                    help='the data initialization type. (uniform/gaussian/gaussian_shift).')
parser.add_argument('--measure', type=str, default="euclidean",
                    help='the distance measure. (euclidean/gaussian.')
parser.add_argument('--Q', type=str, default=1,
                    help='set value of Q in gaussian measure.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

dsts = {
    "CIFAR": datasets.CIFAR100,
    "MNIST": datasets.QMNIST, 
    "Omniglot": datasets.Omniglot,
    "SVHN": datasets.SVHN,
}
dst = dsts[args.data]("~/.torch", download=True)

tp = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

# Convert grayscale to rgb.
if gt_data.shape[0] == 1:
    gt_data = gt_data.repeat(3, 1, 1)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

net = LeNet().to(device)

torch.manual_seed(1234)

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
losses = {
    'iter': [],
    'psnr': [],
    'ssim': [],
    'mse': [],
}
gt_im = gt_data[0].cpu().numpy().transpose((1, 2, 0))

for iters in range(300):

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
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.10f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

        dummy_im = dummy_data[0].cpu().detach().numpy().transpose((1, 2, 0))
        losses['iter'].append(iters)
        losses['psnr'].append(psnr(gt_im, dummy_im))
        losses['mse'].append(mse(gt_im, dummy_im))
        losses['ssim'].append(ssim(gt_im, dummy_im, multichannel=True))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].plot(losses['iter'], losses['mse'])
ax[0].set_title('MSE')
ax[1].plot(losses['iter'], losses['psnr'])
ax[1].set_title('PSNR')
ax[2].plot(losses['iter'], losses['ssim'])
ax[2].set_title('SSIM')
plt.show()

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
