import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def tensor_to_numpy(tensor):
    """Convert tensor to numpy for metrics calculation"""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    return tensor.cpu().numpy()


def calculate_psnr(img1, img2, max_value=1.0):
    """Calculate PSNR between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    img1 = np.clip(img1, 0, max_value)
    img2 = np.clip(img2, 0, max_value)
    
    try:
        return psnr(img1, img2, data_range=max_value)
    except:
        return 0.0


def calculate_ssim(img1, img2, max_value=1.0):
    """Calculate SSIM between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_numpy(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor_to_numpy(img2)
    
    img1 = np.clip(img1, 0, max_value)
    img2 = np.clip(img2, 0, max_value)
    
    try:
        return ssim(img1, img2, data_range=max_value, channel_axis=2)
    except:
        return 0.0


class SSIMLoss(nn.Module):
    """SSIM Loss function"""
    
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                             for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = torch.nn.functional.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel)


class CombinedLoss(nn.Module):
    """Combined L1 + SSIM loss"""
    
    def __init__(self, alpha=0.84, beta=0.16):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.alpha * l1 + self.beta * ssim


def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean