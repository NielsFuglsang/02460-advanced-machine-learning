import torch
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt


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
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
        dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    elif inittype == "gaussian_shift":
        dummy_data = torch.normal(mean=0.5, std=0.5, size=gt_data.size()).to(device).requires_grad_(True)
        dummy_label = torch.normal(mean=0.5, std=0.5, size=gt_onehot_label.size()).to(device).requires_grad_(True)
    else:
        raise ValueError("Only keywords 'uniform' and 'gaussian' are accepted for 'inittype'.")
    return dummy_data, dummy_label