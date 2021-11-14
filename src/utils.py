
import itertools

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def label_to_onehot(target, num_classes=100):
    """Convert labels to one hot encoded format."""

    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    """Calculate cross entropy loss for one hot encoded predictions and targets."""

    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def euclidean_measure(original_dy_dx, dummy_dy_dx):
    """Calculate euclidean distance."""
    grad_diff = 0
    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
        grad_diff += ((gx - gy)**2).sum()
    return grad_diff


def gaussian_measure(sigma=10, Q=1):
    """Calculate gaussian kernel distance measure between gradients."""
    def gauss(original_dy_dx, dummy_dy_dx, sigma=sigma, Q=Q):
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy)**2).sum()

        # Plotting the grads shows small variance around 0. Leads to exp(inf) and process fails
        # if sigma < 100:
        #     sigma = 100

        return Q * (1 - torch.exp(-grad_diff / sigma))

    return gauss

def gaussian_measure_adaptive(sigmas, Qs):
    """Calculate gaussian kernel distance measure between gradients."""
    
    def gauss(original_dy_dx, dummy_dy_dx, sigmas=sigmas, Qs=Qs):
        grad_diff = 0
        for i, (gx, gy) in enumerate(zip(dummy_dy_dx, original_dy_dx)):
            euclid = ((gx - gy)**2).sum() / torch.numel(gx)
            exponential = torch.exp(-euclid / sigmas[i])

            grad_diff += Qs[i] * (1 - exponential)

        return grad_diff
        # return Q * (1 - torch.exp(-grad_diff / sigma))

    return gauss
