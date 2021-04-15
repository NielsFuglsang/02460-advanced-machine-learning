import torch
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt
from torchvision import transforms


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def euclidean_measure(original_dy_dx, dummy_dy_dx):
    grad_diff = 0
    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
        grad_diff += ((gx - gy)**2).sum()
    return grad_diff


def gaussian_measure(sigma=10, Q=1):
    def gauss(original_dy_dx, dummy_dy_dx, sigma=sigma, Q=Q):
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy)**2).sum()

        #Plotting the grads shows small variance around 0. Leads to exp(inf) and process fails
        if sigma < 100:
            sigma = 100

        return Q * (1 - torch.exp(-grad_diff / sigma))

    return gauss


def create_loss_measure(args, original_dy_dx):
    if args.measure == "euclidean":
        return euclidean_measure
    elif args.measure == "gaussian":
        all_grads = [torch.flatten(grad) for grad in original_dy_dx]
        sigma = torch.var(torch.cat(all_grads), dim=0).item()
        return gaussian_measure(sigma=sigma, Q=args.Q)
    else:
        raise ValueError("Only keywords 'euclidean' and 'gaussian' are accepted for 'measure'.")


def init_data(gt_data, gt_onehot_label, device, inittype="uniform"):
    if inittype == "uniform":
        dummy_data = torch.rand(gt_data.size()).to(device).requires_grad_(True)
        dummy_label = torch.rand(gt_onehot_label.size()).to(device).requires_grad_(True)
    elif inittype == "gaussian":
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)*0.1 + 0.5
        dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)*0.1 + 0.5
    else:
        raise ValueError("Only keywords 'uniform' and 'gaussian' are accepted for 'inittype'.")
    return dummy_data, dummy_label

def format_image(dst, index, device):
    """Format CIFAR image to tensor."""
    tp = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])

    gt_data = tp(dst[index][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    return gt_data

def format_label(dst, index, device):
    """Format CIFAR label to tensor"""
    gt_label = torch.Tensor([dst[index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)
    return gt_onehot_label

def make_reconstruction_plots(
        history,
        batch_size,
        val_size,
        indices,
        dst,
        figsize=(12, 8)
    ):
    if batch_size > 1:
        fig, axes = plt.subplots(len(history) + 1, batch_size, figsize=figsize)
        for i in range(len(history)):
            for j in range(batch_size):
                axes[i][j].imshow(history[i][j])
                axes[i][j].set_title(f"it={i * val_size}")
                axes[i][j].axis('off')

        for j in range(batch_size):
            axes[i+1][j].imshow((dst[indices[j]][0]))
            axes[i+1][j].set_title(f"Ground truth. Image {indices[j]}")
            axes[i+1][j].axis('off')
    else:
        fig, axes = plt.subplots(len(history) + 1, 1, figsize=figsize)

        for i in range(len(history)):
            axes[i].imshow(history[i][0])
            axes[i].set_title(f"it={i * val_size}")
            axes[i].axis('off')

        axes[-1].imshow((dst[indices[0]][0]))
        axes[-1].set_title(f"Ground truth. Image {indices[0]}")
        axes[-1].axis('off')

    fig.tight_layout()
    plt.show()
