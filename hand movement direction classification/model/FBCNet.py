import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from config import batch_size

spatial_kernel_num = config.the_spatial_kernel_num
variance_window = config.variance_window

class FBCNet(nn.Module):
    def __init__(self, n_channels, n_time, n_class):
        """
        FBCNet(B, Nb, C, T) -> (B, n_class)
        :param n_channels: the number of input channels
        :param n_time: the length of the time series
        :param n_class: the number of classes
        """
        super(FBCNet, self).__init__()

        # 1. spatial convolution
        self.spatial_conv = nn.Conv2d(9, spatial_kernel_num*9, kernel_size=(n_channels, 1), stride=1, groups=9)
        self.bn = nn.BatchNorm2d(spatial_kernel_num*9)

        # 3. classifier
        self.mlp = nn.Sequential(nn.Flatten(),
                                 nn.Linear(spatial_kernel_num*9*(n_time//variance_window), n_class))

    def forward(self, x):

        # x: (B, Nb, C, T)
        # 1. spatial convolution
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = F.elu(x)

        # x: (B, m*Nb, 1, T)
        # split windows in T dimension
        x_unfold = x.unfold(dimension=-1, size=variance_window, step=variance_window)

        # x_unfold: (B, m*Nb, 1, T//variance_window, variance_window)
        # 2. temporal variance layer
        mean = x_unfold.mean(dim=-1, keepdim=True)
        var = ((x_unfold - mean).pow(2).mean(dim=-1))
        var = torch.log(var + 1e-6)

        # var: (B, m*Nb, 1, T//variance_window)
        # 3. classifier
        x = self.mlp(var)
        return x

if __name__ == '__main__':

    # test the model
    batch_size = config.batch_size
    n_channel = 10
    n_time = 1000
    n_class = 4

    x = torch.randn(batch_size, 9, n_channel, n_time, device=config.device)
    model = FBCNet(n_channels=n_channel, n_time=n_time, n_class=n_class)
    model.to(config.device)
    y = model(x)
    print(y.shape)
