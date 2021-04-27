import pickle

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

dst = datasets.CIFAR100("~/.torch", download=True)

def load(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

ge = load('cur_results/CIFAR_gaussian_euclidean_100_101_212104_161049')
gg = load('cur_results/CIFAR_gaussian_gaussian_100_101_212304_010053_1000')
ges = load('cur_results/CIFAR_gaussian_shift_euclidean_100_101_212104_145343')
ggs = load('cur_results/CIFAR_gaussian_shift_gaussian_100_101_212204_231841_1000')
ue = load('cur_results/CIFAR_uniform_euclidean_100_101_212104_140742')
ug = load('cur_results/CIFAR_uniform_gaussian_100_101_212204_220613_1000')

n =  ue['params']['num_epochs']
iters = range(n)
non_converging = []
for data in [ge, gg, ges, ggs, ue, ug]:
    print(data['params']['lr'])
    for loss in ['mse', 'ssim', 'psnr']:
        l = np.array(data['losses'][loss])
        # Remove non-converging.
        if loss == 'mse':
            mask = (l[:,-1] < l[:,0]) * (l[:,-1] < 2)
            print(np.sum(~mask))
            non_converging.append(np.where(~mask)[0])
        l_removed = l[mask]
        data[loss+'_avg'] = np.mean(l_removed, axis=0)
        data[loss+'_std'] = np.std(l_removed, axis=0)

names = ['Gaussian-Euclidean', 'Gaussian-Gaussian', 'GuassianShift-Euclidean', 'GaussianShift-Gaussian', 'Uniform-Euclidean', 'Uniform-Gaussian']

for i, idxs in enumerate(non_converging):
    fig, axes = plt.subplots(1, len(idxs), figsize=(3*len(idxs), 2))
    fig.suptitle('Non-converging examples for ' + names[i])
    for j, idx in enumerate(idxs):
        axes[j].imshow(dst[idx][0])
    plt.savefig(f'non_converging/{names[i]}.pdf', bbox_inches='tight')