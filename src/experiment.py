from datetime import datetime
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import torch
import torch.nn.functional as F
from torchvision import models, datasets, transforms

from .models import LeNet, weights_init, ResNet18
from .utils import label_to_onehot, cross_entropy_for_onehot, euclidean_measure, gaussian_measure, gaussian_measure_adaptive


class MinMaxScalerVectorized(object):
    """
    If size is (batch_size, channels, n, m), then transforms each channel to the range [0, 1].
    If size is (batch_size, n), then transform each bacth to the range [0, 1].
    """
    def __call__(self, tensor):
        if len(tensor.shape) == 4:
            dim = (2,3)
        else:
            dim = (1)
        dist = (tensor.amax(dim=dim, keepdim=True) - tensor.amin(dim=dim, keepdim=True))
        dist[dist==0.] = 1. # Avoid dividing by zero.
        scale = 1.0 /  dist
        tensor = tensor * scale
        tensor = tensor - tensor.amin(dim=dim, keepdim=True)
        return tensor

class Experiment:
    """Class for running experiments of DLG algorithm given a set parameters (dictionary)."""

    def __init__(self, params=None, rand_ims=False, verbose=True):
        torch.manual_seed(1234)
        # Identify device for computations.
        self.rand_ims = rand_ims
        self.verbose = verbose

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        print("Running on %s" % self.device)

        # Load input parameters.
        self.params = params
        if self.params:
            self.init_with_params()
    
    def init_with_params(self):
        self.set_params()
        # Load dataset.
        self.dst = self.load_dataset()

        # Transforms.
        self.tp = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
        self.tt = transforms.ToPILImage()

        # Specify indices.
        self.random = self.rand_ims
        if self.random:
            self.indices = random.choices(list(range(len(self.dst))), k=self.batch_size)
        else:
            self.indices = np.arange(self.params["index"], self.params["index"] + self.batch_size)

        # Load ground truth data and find gradients.
        self.gt_data, self.gt_label, self.gt_onehot_label = self.load_ground_truths()
        self.inp_channels = self.gt_data.shape[1]

        # Initialize network and compute gradients.
        if self.params["nn"] == 'LeNet':
            self.net = LeNet(self.inp_channels).to(self.device)
        elif self.params["nn"] == 'ResNet':
            self.net = ResNet18(self.inp_channels).to(self.device)
        else:
            print("Model must be given.")

        if self.params["optimizer"] == 'AdamW':
            self.optimizer = torch.optim.AdamW
        else:
            self.optimizer = torch.optim.LBFGS

        self.net.apply(weights_init)
        self.original_dy_dx = self.compute_original_grad()

        # Create loss measure (euclidean or gaussian).
        self.loss_measure = self.create_loss_measure()

        # Training losses and image history.
        self.iters = np.arange(0, self.num_epochs, self.val_size)
        self.losses = {'psnr': [], 'ssim': [], 'mse': []}
        self.history = []
        self.used_indices = []

    def set_params(self):
        """Set all params if params provided on initialization."""
        self.num_epochs = self.params["num_epochs"]
        self.batch_size = self.params["batch_size"]
        self.measure = self.params["measure"]
        self.data_name = self.params["data"]
        self.img_index = self.params["index"]
        self.init_type = self.params["init_type"]
        self.Q = self.params["Q"]
        self.val_size = self.params["val_size"]
        self.n_repeats = self.params["n_repeats"]
        self.lr = self.params["lr"]
        self.sigma = self.params.get("sigma")
        self.idlg = self.params.get("idlg")
                
    def reset(self):
        """Reset network weights and ground truth data."""

        # Reset weights.
        self.net.apply(weights_init)

        # Specify indices.
        if self.random:
            self.indices = random.choices(list(range(len(self.dst))), k=self.batch_size)
        else:
            # Add to previous indeces, to keep same indeces when comparing methods.
            self.indices += self.batch_size

        # Load ground truth data and find gradients.
        self.gt_data, self.gt_label, self.gt_onehot_label = self.load_ground_truths()
        self.original_dy_dx = self.compute_original_grad()

        # Create loss measure (euclidean or gaussian).
        self.loss_measure = self.create_loss_measure()

    def run_multiple(self):
        """Run training on multiple images to get an estimate of performance."""
        print("Image", 0)
        self.train()
        for i in range(self.n_repeats - 1):
            print("Image", i)
            # Reset and run training again.
            self.reset()
            self.train()

    def train(self):
        """Train our network based on the DLG algorithm."""

        dummy_data, dummy_label = self.init_data()
        optimizer = self.optimizer([dummy_data, dummy_label], lr=self.lr)

        gt_im = self.gt_data[0].cpu().numpy().transpose((1, 2, 0))

        train_history = []
        train_loss = {'psnr': [], 'ssim': [], 'mse': []}
        for iters in range(self.num_epochs):
            def closure():
                optimizer.zero_grad()

                dummy_pred = self.net(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net.parameters(), create_graph=True)

                grad_diff = self.loss_measure(self.original_dy_dx, dummy_dy_dx)

                grad_diff.backward()

                return grad_diff

            optimizer.step(closure)
            if iters % self.val_size == 0:
                current_loss = closure()
                if self.verbose:
                    print(iters, "%.10f" % current_loss.item(), flush=True)
                train_history.append([self.tt(dummy_data[i].cpu()) for i in range(self.batch_size)])

                dummy_im = dummy_data[0].cpu().detach().numpy().transpose((1, 2, 0))
                train_loss['psnr'].append(psnr(gt_im, dummy_im))
                train_loss['mse'].append(mse(gt_im, dummy_im))
                train_loss['ssim'].append(ssim(gt_im, dummy_im, multichannel=True))

        # Append training to global variables.
        self.history.append(train_history)
        self.losses['psnr'].append(train_loss['psnr'])
        self.losses['mse'].append(train_loss['mse'])
        self.losses['ssim'].append(train_loss['ssim'])
        self.used_indices.append(self.indices.copy())

    def compute_original_grad(self):
        """Compute original gradients for ground truth data."""
        pred = self.net(self.gt_data)
        y = cross_entropy_for_onehot(pred, self.gt_onehot_label)
        dy_dx = torch.autograd.grad(y, self.net.parameters())

        return list((_.detach().clone() for _ in dy_dx))

    def load_ground_truths(self):
        """Load ground truths from dataset."""

        # Get ground truth batch of images and labels.
        images = [self.format_image(idx) for idx in self.indices]
        gt_label = [self.format_label(idx) for idx in self.indices]
        gt_data = torch.cat(images, 0)
        gt_onehot_label = torch.cat(gt_label, 0)

        return gt_data, gt_label, gt_onehot_label

    def load_dataset(self):
        """Load dataset from torch."""
        dsts = {
            "CIFAR": datasets.CIFAR100,
            "MNIST": datasets.QMNIST,
            "Omniglot": datasets.Omniglot,
            "SVHN": datasets.SVHN,
        }
        return dsts[self.data_name]("~/.torch", download=True)

    def create_loss_measure(self):
        """Create loss measure, either euclidean distance or gaussian kernel."""
        if self.measure == "euclidean":
            return euclidean_measure
        elif self.measure == "gaussian":
            all_grads = [torch.flatten(grad) for grad in self.original_dy_dx]
            if self.sigma:
                sigma = self.sigma
            else:
                sigma = torch.var(torch.cat(all_grads), dim=0).item()

            return gaussian_measure(sigma=sigma, Q=self.Q)
        elif self.measure == "gaussian_adaptive":
            # Calculate sigmas per layer.
            sigmas = [torch.var(grad) for grad in self.original_dy_dx]
            # Put more weight on layers close to the input.
            Qs = [1/(i+1) for i in range(len(self.original_dy_dx))]
            return gaussian_measure_adaptive(sigmas=sigmas, Qs=Qs)
        else:
            raise ValueError(
                "Only keywords 'euclidean', 'gaussian', and 'gaussian_adaptive' are accepted for 'measure'.")

    def init_data(self):
        """Initialize dummy data and label based on parameters."""
        if self.init_type == "uniform":
            dummy_data = torch.rand(self.gt_data.size()).to(
                self.device).requires_grad_(True)
            dummy_label = torch.rand(self.gt_onehot_label.size()).to(
                self.device).requires_grad_(True)
        elif self.init_type == "gaussian":
            dummy_data = torch.randn(self.gt_data.size()).to(
                self.device).requires_grad_(True)
            dummy_label = torch.randn(self.gt_onehot_label.size()).to(
                self.device).requires_grad_(True)
        elif self.init_type == "gaussian_shift":
            dummy_data = torch.normal(mean=0.5, std=0.5, size=self.gt_data.size()).to(
                self.device).requires_grad_(True)
            dummy_label = torch.normal(mean=0.5, std=0.5, size=self.gt_onehot_label.size()).to(
                self.device).requires_grad_(True)
        elif self.init_type == "gaussian_shift2":
            dummy_data = torch.randn(self.gt_data.size())
            dummy_label = torch.randn(self.gt_onehot_label.size())
            scaler = MinMaxScalerVectorized()
            dummy_data = scaler(dummy_data).to(self.device).requires_grad_(True)
            dummy_label = scaler(dummy_label).to(self.device).requires_grad_(True)
        else:
            raise ValueError(
                "Only keywords 'uniform', 'gaussian' and 'gaussian_shift are accepted for 'init_type'.")

        if self.idlg:
            # Use iDLG initialization of dummy label.
            dummy_label = torch.zeros(self.gt_onehot_label.size()).to(self.device).requires_grad_(True)
            with torch.no_grad():
                dummy_label[0, torch.argmax(self.original_dy_dx[-1] * self.original_dy_dx[-1])] = 1

        return dummy_data, dummy_label

    def make_reconstruction_plots(self, filename=None, train_id=0, figsize=(12, 8)):
        """Make reconstruction plots from what was stored in history."""
        ims = self.history[train_id]

        fig, axes = plt.subplots(
            self.batch_size, len(ims) + 1, figsize=figsize)

        if self.batch_size > 1:
            for i in range(self.batch_size):
                for j in range(len(ims)):
                    axes[i][j].imshow(ims[j][i])
                    axes[i][j].set_title(f"it={j * self.val_size}")
                    axes[i][j].axis('off')

            for i in range(self.batch_size):
                axes[i][j + 1].imshow((self.dst[self.used_indices[train_id][i]][0]))
                axes[i][j + 1].set_title(f"Ground truth. Image {self.used_indices[train_id][i]}")
                axes[i][j+1].axis('off')
        else:
            for i in range(len(ims)):
                axes[i].imshow(ims[i][0])
                # axes[i].set_title(f"it={i * self.val_size}")
                axes[i].axis('off')
            axes[i+1].imshow((self.dst[self.used_indices[train_id][0]][0]))
            # axes[i + 1].set_title(f"True img {self.used_indices[train_id][0]}")
            axes[i+1].axis('off')
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        else:
            fig.tight_layout()
            plt.show()

    def format_image(self, index):
        """Format image to tensor."""
        gt_data = self.tp(self.dst[index][0]).to(self.device)

        gt_data = gt_data.view(1, *gt_data.size())

        return gt_data

    def format_label(self, index):
        """Format label to tensor."""
        gt_label = torch.Tensor([self.dst[index][1]]).long().to(self.device)
        gt_label = gt_label.view(1, )
        gt_onehot_label = label_to_onehot(gt_label)

        return gt_onehot_label

    
    def save_experiment(self):
        """Save the results of an experiment in a pickle file."""
        results = {
            "params": self.params,
            "losses": self.losses,
            "history": self.history,
            "used_indices": self.used_indices
        }
        
        now = datetime.now()
        # _{now.strftime(''%Y%d%m_%H%M%S')}
        filename = "./results/{}_{}_{}_{}_{}_{}{}".format(
            self.params['data'],
            self.params['init_type'],
            self.params['measure'],
            self.params['n_repeats'],
            self.params['num_epochs'],
            now.strftime('%y%d%m_%H%M%S'),
            '_' + str(self.sigma) if self.sigma else ""
        )

        with open(filename,'wb') as f:
            pickle.dump(results, f)
        
    def load_experiment(self, pickle_file):
        """Load a previous experiment. Can be used for later data processing and reconstruction plots."""
        
        with open(pickle_file, "rb") as f:
            results = pickle.load(f)
        
        self.params = results["params"]
        self.init_with_params()
        self.losses = results["losses"]
        self.history = results["history"]
        self.used_indices = results["used_indices"]