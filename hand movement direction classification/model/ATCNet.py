import torch
import torch.nn as nn
import torch.nn.functional as F
import config

F1 = config.temporal_conv_kernel_num
sampling_rate = config.sampling_rate
D = config.channel_kernel_num
P2 = config.stride_of_avg_pool2
n_head = config.n_head
dropout_rate = config.dropout_rate
n_windows = config.n_windows
tcb_residual_num = config.tcb_residual_num
tcb_kernel_size = config.tcb_kernel_size

class ConvolutionalBlock(nn.Module):
    def __init__(self, n_channels):
        """
        the convolutional block (B, C, T) -> (B, Tc, D*F1)
        :param n_channels: the number of channels
        """
        super(ConvolutionalBlock, self).__init__()

        # 1. Temporal conv
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, sampling_rate//4), stride=(1, 1), padding=(0, (sampling_rate//4)//2))
        self.bn1 = nn.BatchNorm2d(F1)

        # 2. Depthwise channel conv
        self.depthwise_channel_conv = nn.Conv2d(in_channels=F1, out_channels=D*F1, kernel_size=(n_channels, 1), groups=F1)
        self.bn2 = nn.BatchNorm2d(D*F1)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

        # 3. Spatial conv
        self.spatial_conv = nn.Conv2d(in_channels=D*F1, out_channels=D*F1, kernel_size=(1, 16), stride=(1, 1), padding=(0, 8))
        self.bn3 = nn.BatchNorm2d(D*F1)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, P2), stride=(1, P2))

    def forward(self, x):

        # x: (B, C, T)
        x = x.unsqueeze(1)

        # x: (B, 1, C, T)
        # 1. Temporal conv
        x = self.temporal_conv(x)
        x = self.bn1(x)

        # x: (B, F1, C, T+1)
        # 2. Depthwise channel conv
        x = self.depthwise_channel_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.avg_pool1(x)

        # x: (B, D*F1, 1, (T+1)/8)
        # 3. Spatial conv
        x = self.spatial_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.avg_pool2(x)

        # x: (B, D*F1, 1, Tc), Tc=(T+1)/(8*P2)
        x = x.squeeze(2)
        x = x.transpose(1, 2)

        # x: (B, Tc, D*F1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dilation):
        super(ResidualBlock, self).__init__()

        # 1. causal dilated conv
        self.causal_dilated_conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=tcb_kernel_size, stride=1, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(32)
        self.causal_dilated_conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=tcb_kernel_size, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):

        # x: (B, 32, Tw)
        # 1. causal dilated conv
        out = self.causal_dilated_conv1(x)
        out = self.bn1(out)
        out = F.elu(out)
        out = self.causal_dilated_conv2(out)
        out = self.bn2(out)
        out = F.elu(out)
        x = out+x[:, :, -out.shape[-1]:]
        x = F.elu(x)

        # x: (B, 32, Tw')
        return x

class TCB(nn.Module):
    def __init__(self):
        super(TCB, self).__init__()

        if D*F1 != 32:
            self.conv = nn.Conv1d(in_channels=D*F1, out_channels=32, kernel_size=1, stride=1)

        # 1. causal dilated conv
        self.causal_dilated_conv = nn.Sequential(*[ResidualBlock(dilation=2**d) for d in range(tcb_residual_num)])

    def forward(self, x):

        # x: (B, D*F1, Tw)
        if D*F1 != 32:
            x = self.conv(x)
        _, _, T = x.shape

        # x: (B, 32, Tw)
        rfs = 1+2*(tcb_kernel_size-1)*(2**tcb_residual_num-1)
        padding_size = rfs-T
        if padding_size < 0:
            raise ValueError("Padding size cannot be negative, please check the TCB's input size")
        x = F.pad(x, (padding_size, 0)) # only pad the left side

        # x: (B, 32, rfs)
        # 1. causal dilated conv
        x = self.causal_dilated_conv(x)

        # x: (B, 32, 1)
        x = x.squeeze(2)

        # x: (B, 32)
        return x

class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()

        # 2. Multi-head self-attention
        self.multi_head_self_attention = nn.MultiheadAttention(embed_dim=D*F1, num_heads=n_head, batch_first=True, dropout=0.5)

        # 3. Temporal convolutional block
        self.temporal_conv = TCB()

    def forward(self, x):

        # x: (B, Tc, D*F1)
        # 1. Convolutional based sliding window
        _, T, E = x.shape
        x_list = []
        for i in range(n_windows):
            t_window = T-n_windows+1
            x_slided = x[:, i:i+t_window, :]

            # x_slided: (B, Tw, D*F1)
            # 2. Multi-head self-attention
            out, _ = self.multi_head_self_attention(x_slided, x_slided, x_slided)
            x_slided = out+x_slided

            # x_slided: (B, Tw, D*F1)
            # 3. Temporal convolutional block
            x_slided = x_slided.transpose(1, 2)
            x_slided = self.temporal_conv(x_slided)

            # x_slided: (B, 32)
            x_list.append(x_slided)
        x = torch.cat(x_list, 1)

        # x: (B, 32*n_windows)
        return x

class ATCNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        ATCNet (B, C, T) -> (B, n_classes)
        :param n_channels: the number of input channels
        :param n_classes: the number of classes
        """
        super(ATCNet, self).__init__()

        # 1. Convolutional block
        self.conv = ConvolutionalBlock(n_channels=n_channels)

        # 2. Transformer block
        self.transformers = TransformerBlock()

        # 3. Classifier
        self.mlp = nn.Linear(32*n_windows, n_classes)

    def forward(self, x):

        # x: (B, C, T)
        x = self.conv(x)

        # x: (B, C, Tc)
        x = self.transformers(x)

        # x: (B, 32*n_windows)
        x = self.mlp(x)

        # x: (B, n_classes)
        return x

if __name__ == '__main__':

    # test the model
    x = torch.rand(config.batch_size, 22, 1000, device=config.device)
    model = ATCNet(n_channels=22, n_classes=4)
    model.to(config.device)
    y = model(x)
    print(y.shape)
