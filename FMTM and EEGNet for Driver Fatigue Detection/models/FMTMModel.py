import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import config

kernel_num = config.kernel_num
sampling_rate = config.sampling_rate
alpha_list = config.alpha_list
batch_size = config.batch_size
electrode_num = config.electrode_num
feature_num = config.feature_num
graph_hidden_size = config.graph_hidden_size
n_class = config.n_class
partitions = config.partitions
attention_hidden_size = config.attention_hidden_size
region_num = len(partitions)
dropout_rate = config.dropout_rate
device = config.device

class Preprocessor(nn.Module):
    def __init__(self):
        super(Preprocessor, self).__init__()

        # the multiscale convolution layer
        self.conv_layer = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=kernel_num,
                                              kernel_size=(1, int(alpha*sampling_rate)),
                                              stride=(1, 1)) for alpha in alpha_list])
        self.batch_norm = nn.BatchNorm2d(num_features=kernel_num)

        # the average pooling layer
        self.pool_layer = nn.AvgPool2d(kernel_size=(1, 64), stride=(1, 32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_list = []
        for conv in self.conv_layer: # data goes through three kinds of conv kernels in parallel
            out = conv(x) # (B, 1, C, L) -> (B, T, C, Fk)
            out = self.batch_norm(out) # (B, T, C, Fk) -> (B, T, C, Fk)
            out = torch.pow(out, 2) # (B, T, C, Fk) -> (B, T, C, Fk)
            out = self.pool_layer(out) # (B, T, C, Fk) -> (B, T, C, Fk')
            out = torch.log(out) # (B, T, C, Fk') -> (B, T, C, Fk')
            x_list.append(out)
        x = torch.cat(x_list, dim=-1) # cat(B, T, C, Fk') -> (B, T, C, F')
        x = torch.mean(x, dim=1) # (B, T, C, F') -> (B, C, F')
        return x

class GCN(nn.Module):
    def __init__(self, channel_num: int):
        """
        :param channel_num: the number of channel, when it's in the case of micro channel_num = C. Otherwise, channel_num = M
        """
        super(GCN, self).__init__()

        # BatchNormalize
        self.batch_norm1 = nn.BatchNorm1d(num_features=channel_num)

        # MLP-based weights
        self.weight_graph = nn.Parameter(torch.Tensor(feature_num, graph_hidden_size))
        self.bias_graph = nn.Parameter(torch.Tensor(graph_hidden_size))

        # graph adjacent matrix weight
        self.adjacent_matrix_weight = nn.Parameter(torch.Tensor(1, feature_num))

        # MLP
        self.flatten = nn.Flatten()
        self.MLP = nn.Sequential(nn.Linear(in_features=channel_num*graph_hidden_size, out_features=4*channel_num*graph_hidden_size),
                                 nn.BatchNorm1d(num_features=4*channel_num*graph_hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(p=dropout_rate),
                                 nn.Linear(in_features=4*channel_num*graph_hidden_size, out_features=channel_num*graph_hidden_size),
                                 nn.BatchNorm1d(num_features=channel_num*graph_hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.MLPs = nn.Sequential(*[self.MLP for _ in range(3)])
        self.final_linear = nn.Linear(in_features=channel_num*graph_hidden_size, out_features=n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get the input's shape
        B, C, _ = x.shape

        # Calculate the graph adjacent matrix
        A_graph = torch.empty(size=[B, C, C], device=device) # (B, C, C)
        for b in range(B):
            for m in range(C):
                for n in range(C):
                    A_graph[b, m, n] = torch.exp(F.relu(self.adjacent_matrix_weight @ torch.abs(x[b, m, :] - x[b, n, :])))
        A_graph = A_graph/torch.sum(A_graph, dim=-1, keepdim=True)
        A_graph = utils.laplacian_norm_batch(A_graph) # (B, C, C) -> (B, C, C)

        # GNN forward process
        x = self.batch_norm1(x) # (B, C, F') -> (B, C, F')
        x = torch.matmul(x, self.weight_graph)+self.bias_graph # (B, C, F') -> (B, C, h)
        x = A_graph@x # (B, C, C) @ (B, C, h) -> (B, C, h)
        x = F.relu(x) # (B, C, h) -> (B, C, h)

        # MLP forward process
        x = self.flatten(x) # (B, C, h) -> (B, C*h)
        x = self.MLPs(x) # (B, C*h) -> (B, C*h)
        x = self.final_linear(x) # (B, C*h) -> (B, n_class)
        return x

class FromMicroToMeso(nn.Module):
    def __init__(self):
        super(FromMicroToMeso, self).__init__()
        self.attention_weight = nn.Parameter(torch.Tensor(feature_num, attention_hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # divide the input data based on dividing rule
        z_list = [x[:, idxs, :] for idxs in partitions] # (B, C, F') -> List((B, Ni, F')) the length of List is M

        # consider the regions one by one
        output = []
        for z in z_list:

            # self-attention based mechanism
            temp = z@self.attention_weight # (B, Ni, F') @ (F', d) -> (B, Ni, d)
            ei = (temp)@(torch.transpose(temp, 1, 2)) # (B, Ni, d) @ (B, d, Ni) -> (B, Ni, Ni)
            ei = F.leaky_relu(ei) # (B, Ni, Ni) -> (B, Ni, Ni)
            Lambda = torch.unsqueeze(torch.sum(ei, dim=1), dim=1) # (B, Ni, Ni) -> (B, 1, Ni)
            z = F.softmax(Lambda, dim=-1)@z # (B, 1, Ni) @ (B, Ni, F') -> (B, 1, F')
            output.append(z)

        # concat the result from each region together
        output = torch.concat(output, dim=1) # List((B, 1, F')) -> (B, M, F')
        return output


class FMTMModel(nn.Module):
    def __init__(self, is_micro = True):
        super(FMTMModel, self).__init__()
        self.preprocessor = Preprocessor()
        self.is_micro = is_micro
        if is_micro:
            self.gcn = GCN(channel_num=electrode_num)
        else:
            self.from_micro_to_meso = FromMicroToMeso()
            self.gcn = GCN(channel_num=region_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessor(x) # (B, 1, C, L) -> (B, C, F')
        if not self.is_micro:
            x = self.from_micro_to_meso(x) # (B, C, F') -> (B, M, F')
        x = self.gcn(x) # (B, C(M), F') -> (B, n_class)
        return x

if __name__ == "__main__":

    # test the model
    x = torch.rand(batch_size, 1, electrode_num, 384, device=device)
    y = torch.zeros(batch_size, device=device, dtype=torch.long)
    fmtm_model = FMTMModel(is_micro=True)
    fmtm_model.to(device)
    logits = fmtm_model(x)
    loss = F.cross_entropy(logits, y)
    print(loss.item())
    print(logits.shape)