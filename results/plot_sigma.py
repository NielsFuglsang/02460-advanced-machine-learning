import pickle

import matplotlib.pyplot as plt
import numpy as np

def load(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

s1 = load('sigma_results/cur/CIFAR_uniform_gaussian_100_101_212204_210141_1')
s2 = load('sigma_results/cur/CIFAR_uniform_gaussian_100_101_212204_210346_10')
s3 = load('sigma_results/cur/CIFAR_uniform_gaussian_100_101_212204_210750_50')
s4 = load('sigma_results/cur/CIFAR_uniform_gaussian_100_101_212204_212416_100')
s5 = load('sigma_results/cur/CIFAR_uniform_gaussian_100_101_212204_214007_150')
s6 = load('sigma_results/cur/CIFAR_uniform_gaussian_100_101_212204_215429_200')
s7 = load('sigma_results/cur/CIFAR_uniform_gaussian_100_101_212204_220613_1000')

n =  s1['params']['num_epochs']
iters = range(n)
non_converging = []
for data in [s1, s2, s3, s4, s5, s6, s7]:
    print(data['params']['nn'])
    for loss in ['mse', 'ssim', 'psnr']:
        l = np.array(data['losses'][loss])
        # Remove non-converging.
        if loss == 'mse':
            mask = (l[:,-1] < l[:,0]) * (l[:,-1] < 2)
            print(np.sum(~mask))
            non_converging.append(np.sum(~mask))

        l_removed = l[mask]
        if len(l_removed) > 0:
            data[loss+'_avg'] = np.mean(l_removed, axis=0)
            data[loss+'_std'] = np.std(l_removed, axis=0)
        else:
            data[loss+'_avg'] = np.zeros(n)
            data[loss+'_std'] = np.zeros(n)
print(non_converging)
# fig, ax = plt.subplots(1,1, figsize=(10,5))
# ax.plot(range(len(mse[0])), mse_avg)
# ax.grid()
# ax.set_xlabel('Iterations x 100')
# ax.set_ylabel('Mean squared error')
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f"Averaging over {n} CIFAR images. LR: {s1['params']['lr']}")


# axes[0].plot(iters, ge['mse_avg'], label='gaussian-euclidean')
# axes[0].plot(iters, gg['mse_avg'], label='gaussian-gaussian')
axes[0].plot(iters, s4['mse_avg'], label='sigma = 100')
axes[0].plot(iters, s5['mse_avg'], label='sigma = 150')
axes[0].plot(iters, s6['mse_avg'], label='sigma = 200')
axes[0].plot(iters, s7['mse_avg'], label='sigma = 1000')
# axes[0].set_yscale('log')
axes[0].legend()
axes[0].set_ylabel('MSE')
axes[0].set_xlabel('Iterations')
axes[0].grid(True)

# axes[1].plot(iters, ge['ssim_avg'], label='gaussian-euclidean')
# axes[1].plot(iters, gg['ssim_avg'], label='gaussian-gaussian')
axes[1].plot(iters, s4['ssim_avg'], label='sigma = 100')
axes[1].plot(iters, s5['ssim_avg'], label='sigma = 150')
axes[1].plot(iters, s6['ssim_avg'], label='sigma = 200')
axes[1].plot(iters, s7['ssim_avg'], label='sigma = 1000')
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