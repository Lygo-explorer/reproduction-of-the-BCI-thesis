import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import config

kernel_num = config.kernel_num
sampling_rate = config.sampling_rate
alpha_list = config.alpha_list
batch_size = config.batch_size
channel_num = config.channel_num
feature_num = config.feature_num
hidden_size = config.hidden_size
n_class = config.n_class

class Preprocessor(nn.Module):
    def __init__(self):
        super(Preprocessor, self).__init__()

        # the multiscale convolution layer
        self.conv_layer = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=kernel_num,
                                              kernel_size=(1, int(alpha*sampling_rate)),
                                              stride=(1, 1)) for alpha in alpha_list])

        # the average pooling layer
        self.pool_layer = nn.AvgPool2d(kernel_size=(1, 64), stride=(1, 32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_list = []
        for conv in self.conv_layer: # data goes through three kinds of conv kernels in parallel
            out = conv(x) # (B, 1, C, L) -> (B, T, C, Fk)
            out = torch.pow(out, 2) # (B, T, C, Fk) -> (B, T, C, Fk)
            out = self.pool_layer(out) # (B, T, C, Fk) -> (B, T, C, Fk')
            out = torch.log(out) # (B, T, C, Fk') -> (B, T, C, Fk')
            x_list.append(out)
        x = torch.cat(x_list, dim=-1) # cat(B, T, C, Fk') -> (B, T, C, F')
        x = torch.mean(x, dim=1) # (B, T, C, F') -> (B, C, F')
        return x

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        # record the basic size parameters
        self.channels = channel_num
        self.batch = batch_size

        # BatchNormalize
        self.batch_norm = nn.BatchNorm1d(num_features=channel_num)

        # MLP-based weights
        self.weight_graph = nn.Parameter(torch.Tensor(feature_num, hidden_size))
        self.bias_graph = nn.Parameter(torch.Tensor(hidden_size))

        # graph adjacent matrix weight
        self.adjacent_matrix_weight = nn.Parameter(torch.Tensor(1, feature_num))

        # MLP
        self.MLP = nn.Sequential(nn.Linear(in_features=channel_num*hidden_size, out_features=4*channel_num*hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(in_features=4*channel_num*hidden_size, out_features=channel_num*hidden_size))
        self.MLPs = nn.Sequential(*[self.MLP for _ in range(3)])
        self.flatten = nn.Flatten()
        self.final_linear = nn.Linear(in_features=channel_num*hidden_size, out_features=n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Calculate the graph adjacent matrix
        A_graph = torch.Tensor(self.batch, self.channels, self.channels) # (B, C, C)
        for b in range(self.batch):
            for m in range(self.channels):
                for n in range(self.channels):
                    A_graph[b, m, n] = torch.exp(F.relu(self.adjacent_matrix_weight @ torch.abs(x[b, m, :] - x[b, n, :])))
        A_graph = A_graph/torch.sum(A_graph, dim=-1, keepdim=True)
        A_graph = utils.laplacian_norm_batch(A_graph) # (B, C, C) -> (B, C, C)

        # forward process
        x = self.batch_norm(x) # (B, C, F') -> (B, C, F')
        x = torch.matmul(x, self.weight_graph)+self.bias_graph # (B, C, F') -> (B, C, h)
        x = A_graph@x # (B, C, h) -> (B, C, h)
        x = F.relu(x) # (B, C, h) -> (B, C, h)
        x = self.flatten(x) # (B, C, h) -> (B, C*h)
        x = self.MLPs(x) # (B, C*h) -> (B, C*h)
        x = self.final_linear(x) # (B, C*h) -> (B, n_class)
        x = F.softmax(x, dim=-1) # (B, n_class) -> (B, n_class)
        return x

class FromMicroToMeso(nn.Module):
    def __init__(self):
        super(FromMicroToMeso, self).__init__()

class FMTMModel(nn.Module):
    def __init__(self):
        super(FMTMModel, self).__init__()
        self.preprocessor = Preprocessor()
        self.gcn = GCN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessor(x)
        x = self.gcn(x)
        return x
