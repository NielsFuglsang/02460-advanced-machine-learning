import pickle

import matplotlib.pyplot as plt
import numpy as np

def load(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

ge = load('cur_results/CIFAR_gaussian_euclidean_100_101_212504_212253')
gg = load('cur_results/CIFAR_gaussian_gaussian_100_101_212504_205224_1000')
ges = load('cur_results/CIFAR_gaussian_shift2_euclidean_100_101_212504_202740')
ggs = load('cur_results/CIFAR_gaussian_shift2_gaussian_100_101_212504_192637_1000')
ue = load('cur_results/CIFAR_uniform_euclidean_100_101_212504_191503')
ug = load('cur_results/CIFAR_uniform_gaussian_100_101_212504_183032_1000')

n =  ue['params']['num_epochs']
iters = range(n)
for data in [ge, gg, ges, ggs, ue, ug]:
    print(data['params']['lr'])
    for loss in ['mse', 'ssim', 'psnr']:
        l = np.array(data['losses'][loss])
        # Remove non-converging.
        if loss == 'mse':
            mask = (l[:,-1] < l[:,0]) * (l[:,-1] < 2)
            print(np.sum(~mask))
        l_removed = l[mask]
        data[loss+'_avg'] = np.mean(l_removed, axis=0)
        data[loss+'_std'] = np.std(l_removed, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f"Averaging over {n} CIFAR images. LR: {ge['params']['lr']}. Sigma = 1000.")


# axes[0].plot(iters, ge['mse_avg'], label='gaussian-euclidean')
# axes[0].plot(iters, gg['mse_avg'], label='gaussian-gaussian')
axes[0].plot(iters, ges['mse_avg'], label='gaussian-shift-euclidean')
axes[0].plot(iters, ggs['mse_avg'], label='gaussian-shift-gaussian')
axes[0].plot(iters, ue['mse_avg'], label='uniform-euclidean')
axes[0].plot(iters, ug['mse_avg'], label='uniform-gaussian')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].set_ylabel('MSE')
axes[0].set_xlabel('Iterations')
axes[0].grid(True)

# axes[1].plot(iters, ge['ssim_avg'], label='gaussian-euclidean')
# axes[1].plot(iters, gg['ssim_avg'], label='gaussian-gaussian')
axes[1].plot(iters, ges['ssim_avg'], label='gaussian-shift-euclidean')
axes[1].plot(iters, ggs['ssim_avg'], label='gaussian-shift-gaussian')
axes[1].plot(iters, ue['ssim_avg'], label='uniform-euclidean')
axes[1].plot(iters, ug['ssim_avg'], label='uniform-gaussian')
axes[1].legend()
axes[1].set_ylabel('SSIM')
axes[1].set_xlabel('Iterations')
axes[1].grid(True)

# axes[2].plot(iters, ge['psnr_avg'], label='gaussian-euclidean')
# axes[2].plot(iters, gg['psnr_avg'], label='gaussian-gaussian')
# axes[2].plot(iters, ue['psnr_avg'], label='uniform-euclidean')
# axes[2].plot(iters, ug['psnr_avg'], label='uniform-gaussian')
# axes[2].legend()
# axes[2].set_ylabel('PSNR')
# axes[2].set_xlabel('Iterations')
# axes[2].grid(True)

plt.tight_layout()
plt.savefig('combined.pdf', bbox_inches='tight')