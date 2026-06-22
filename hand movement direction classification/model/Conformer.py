import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# set up some model parameters
k = config.temporal_conv_out_channel
head_num = config.head_num
dropout_rate = config.dropout_rate
N = config.execution_times
embedding_size = config.embedding_size

class ConvolutionModule(nn.Module):
    def __init__(self, ch):
        """
        Convolution Module, (B, C, T) -> (B, F, embedding_size)
        :param ch: the number of channels
        """
        super(ConvolutionModule, self).__init__()

        # 1. Temporal convolution
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=k, kernel_size=(1,25), stride=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(k)

        # 2. Spatial convolution
        self.spatial_conv = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(ch, 1), stride=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(k)

        # 3. Avg pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1,15))
        self.drop_out = nn.Dropout(p=dropout_rate)

        # 4. projection
        self.projection = nn.Conv2d(k, embedding_size, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):

        # x: (B, C, T)
        x = torch.unsqueeze(x, dim=1)

        # x: (B, 1, C, T)
        # 1. Temporal convolution
        x = self.temporal_conv(x)
        x = self.batch_norm1(x)
        x = F.elu(x)

        # x: (B, k, C, F_temp)
        # 2. Spatial convolution
        x = self.spatial_conv(x)
        x = self.batch_norm2(x)
        x = F.elu(x)

        # x: (B, k, 1, F_temp)
        # 3. Avg pooling
        x = self.avg_pool(x)
        x = self.drop_out(x)

        # x: (B, k, 1, F)
        # 4. projection
        x = self.projection(x)

        # x: (B, embedding_size, 1, F)
        x = torch.squeeze(x)
        x = torch.transpose(x, 1, 2)

        # x: (B, F, embedding_size)
        return x

class SelfAttentionModule(nn.Module):
    def __init__(self, f):
        """
        self attention module, (B, F, embedding_size) -> (B, F, embedding_size//head_num)
        :param f: the convolution layer's output's F in the size of (B, F, embedding_size)
        """
        super(SelfAttentionModule, self).__init__()

        # set the convolution layer's output's F in the size of (B, F, embedding_size)
        self.f = f

        # 1. Q, K, V generation
        self.key = nn.Linear(embedding_size, embedding_size//head_num, bias=False)
        self.query = nn.Linear(embedding_size, embedding_size//head_num, bias=False)
        self.value = nn.Linear(embedding_size, embedding_size//head_num, bias=False)
        # self.register_buffer('mask', torch.tril(torch.ones(f, f)))  # create a constant tensor that won't change during optimizer step.
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        # x: (B, F, embedding_size)
        # 1. Q, K, V generation
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Q: (F, embedding_size//head_num) K: (F, embedding_size//head_num) V: (F, embedding_size//head_num)
        # 2. self-attention block
        x = Q@(torch.transpose(K, 1, 2))
        # x = x.masked_fill(self.mask[:f, :f] == 0, float('-inf'))
        x = F.softmax(x/self.f**0.5, dim=-1)
        x = self.dropout(x)
        x = x @ V

        # x: (B, F, embedding_size//head_num)
        return x

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, f):
        """
        multi head attention module, (B, F, embedding_size) -> (B, F, embedding_size)
        :param f: the convolution layer's output's F in the size of (B, F, embedding_size)
        """
        super(MultiHeadAttentionModule, self).__init__()

        # 1. multi head self-attention
        self.multi_head = nn.ModuleList([SelfAttentionModule(f) for _ in range(head_num)])

        # 2. projection
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        # x: (B, F, embedding_size)
        # 1. multi head self-attention
        x_list = []
        for head in self.multi_head:
            out = head(x)
            x_list.append(out)

        # x per head: (B, F, embedding_size//head_num)
        x = torch.cat(x_list, dim=-1)

        # x: (B, F, embedding_size)
        # 2. projection
        x = self.projection(x)
        x = self.dropout(x)

        # x: (B, F, embedding_size)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, f):
        """
        transformer block, (B, F, embedding_size) -> (B, F, embedding_size)
        :param f: the convolution layer's output's F in the size of (B, F, embedding_size)
        """
        super(TransformerBlock, self).__init__()

        # 1. multi-head self-attention module
        self.layer_norm1 = nn.LayerNorm([f, embedding_size])
        self.attn = MultiHeadAttentionModule(f)

        # 2. feed forward
        self.layer_norm2 = nn.LayerNorm([f, embedding_size])
        self.feed_forward = nn.Sequential(nn.Linear(embedding_size, 4*embedding_size),
                                          nn.GELU(),
                                          nn.Dropout(p=dropout_rate),
                                          nn.Linear(4*embedding_size, embedding_size),
                                          nn.Dropout(p=dropout_rate))

    def forward(self, x):

        # x: (B, F, embedding_size)
        # 1. multi-head self-attention module
        z1 = self.layer_norm1(x)
        x = x+self.attn(z1)

        # x: (B, F, embedding_size)
        z2 = self.layer_norm2(x)
        x = x+self.feed_forward(z2)

        # x: (B, F, embedding_size)
        return x

class Conformer(nn.Module):
    def __init__(self, ch, n_class, n_times):
        """
        conformer module, (B, C, T) -> (B, n_class)
        :param ch: the number of channels
        :param n_class: the number of classes
        :param n_times: the number of time samples
        """
        super(Conformer, self).__init__()

        # 1. convolution module
        self.conv = ConvolutionModule(ch)

        # set the output's F in the size of (B, k, F)
        f = self.conv(torch.rand(config.batch_size, ch, n_times)).shape[1]

        # 2. multi-head attention module
        self.attn = nn.Sequential(*[TransformerBlock(f) for _ in range(N)])

        # 3. classifier
        self.mlp = nn.Sequential(nn.Flatten(),
                                 nn.Linear(f*embedding_size, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ELU(),
                                 nn.Dropout(p=dropout_rate),
                                 nn.Linear(256, 32),
                                 nn.BatchNorm1d(32),
                                 nn.ELU(),
                                 nn.Dropout(p=dropout_rate),
                                 nn.Linear(32, n_class))

    def forward(self, x):

        # x: (B, C, T)
        # 1. convolution module
        x = self.conv(x)

        # x: (B, F, embedding_size)
        # 2. multi-head attention module
        x = self.attn(x)

        # x: (B, F, embedding_size)
        # 3. classifier
        x = self.mlp(x)

        # x: (B, n_class)
        return x

if __name__ == "__main__":

    # set some data parameters
    ch = 3
    n_class = 2
    time_num = 1000

    # test the model
    x = torch.rand(config.batch_size, ch, time_num, device=config.device)
    model = Conformer(ch, n_class, time_num)
    model.to(config.device)
    y = model(x)
    print(y.shape)