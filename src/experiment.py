
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import torch
import torch.nn.functional as F
from torchvision import models, datasets, transforms

from .models import LeNet, weights_init
from .utils import label_to_onehot, cross_entropy_for_onehot, euclidean_measure, gaussian_measure, init_data

class Experiment:

    def __init__(self, params):
        
        # Identify device for computations.
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        print("Running on %s" % self.device)

        # Load input parameters.
        self.num_epochs = params["num_epochs"]
        self.batch_size = params["batch_size"]
        self.measure = params["measure"]
        self.data_name = params["data"]
        self.img_index = params["index"]
        self.init_type = params["init_type"]
        self.Q = params["Q"]

        # Initialize network.
        self.net = LeNet().to(self.device)
        self.net.apply(weights_init)
        
        # Learning rate.
        self.lr = 0.01

        # Transforms.
        self.tp = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
        self.tt = transforms.ToPILImage()

        # Load dataset.
        self.dst = self.load_dataset()

        # Load ground truth data and find gradients.
        self.gt_data, self.gt_label, self.gt_onehot_label = self.load_ground_truths()
        self.original_dy_dx = self.compute_original_grad()

        # Create loss measure (euclidean or gaussian).
        self.loss_measure = self.create_loss_measure()

    def compute_original_grad(self):
        """Compute original gradients for ground truth data."""
        pred = self.net(self.gt_data)
        y = cross_entropy_for_onehot(pred, self.gt_onehot_label)
        dy_dx = torch.autograd.grad(y, self.net.parameters())

        return list((_.detach().clone() for _ in dy_dx))

        
    def train(self):
        """Train our network based on the DLG algorithm."""

        dummy_data, dummy_label = self.init_data()
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.lr)
        
        torch.manual_seed(1234)

        history = []
        losses = {
            'iter': [],
            'psnr': [],
            'ssim': [],
            'mse': [],
        }
        gt_im = self.gt_data[0].cpu().numpy().transpose((1, 2, 0))
        
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
            if iters % 10 == 0:
                current_loss = closure()
                print(iters, "%.10f" % current_loss.item())
                history.append(self.tt(dummy_data[0].cpu()))

                dummy_im = dummy_data[0].cpu().detach().numpy().transpose((1, 2, 0))
                losses['iter'].append(iters)
                losses['psnr'].append(psnr(gt_im, dummy_im))
                losses['mse'].append(mse(gt_im, dummy_im))
                losses['ssim'].append(ssim(gt_im, dummy_im, multichannel=True))
            
        return losses

    def load_ground_truths(self):
        """Load ground truths from dataset."""
        gt_data = self.tp(self.dst[self.img_index][0]).to(self.device)

        # Convert grayscale to rgb.
        if gt_data.shape[0] == 1:
            gt_data = gt_data.repeat(3, 1, 1)

        # if len(params["image"]) > 1:
        #     gt_data = Image.open(params["image"])
        #     gt_data = tp(gt_data).to(self.device)

        gt_data = gt_data.view(1, *gt_data.size())
        gt_label = torch.Tensor([self.dst[self.img_index][1]]).long().to(self.device)
        gt_label = gt_label.view(1, )
        gt_onehot_label = label_to_onehot(gt_label)

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
            sigma = torch.var(torch.cat(all_grads), dim=0).item()
            return gaussian_measure(sigma=sigma, Q=self.Q)
        else:
            raise ValueError("Only keywords 'euclidean' and 'gaussian' are accepted for 'measure'.")

    def init_data(self):
        """Initialize dummy data and label."""
        if self.init_type == "uniform":
            dummy_data = torch.rand(self.gt_data.size()).to(self.device).requires_grad_(True)
            dummy_label = torch.rand(self.gt_onehot_label.size()).to(self.device).requires_grad_(True)
        elif self.init_type == "gaussian":
            dummy_data = torch.randn(self.gt_data.size()).to(self.device).requires_grad_(True)
            dummy_label = torch.randn(self.gt_onehot_label.size()).to(self.device).requires_grad_(True)
        elif self.init_type == "gaussian_shift":
            dummy_data = torch.normal(mean=0.5, std=0.5, size=self.gt_data.size()).to(self.device).requires_grad_(True)
            dummy_label = torch.normal(mean=0.5, std=0.5, size=self.gt_onehot_label.size()).to(self.device).requires_grad_(True)
        else:
            raise ValueError("Only keywords 'uniform', 'gaussian' and 'gaussian_shift are accepted for 'init_type'.")
        return dummy_data, dummy_label