import torch
import torch.nn as nn
import torch.nn.functional as F
import config
n_channels = config.electrode_num
n_times = config.time_num
n_classes = config.n_class
F1 = config.temporal_kernel_num
D = config.spatial_kernel_num
F2 = config.separable_kernel_num
dropout = config.dropout_rate

class EEGNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        # =========================
        # 1. Temporal Convolution
        # =========================
        self.temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, 64),
            padding=(0, 32),
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(F1)

        # =========================
        # 2. Spatial Depthwise Conv
        # =========================
        self.spatial = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(F1 * D)

        # =========================
        # 3. Separable Conv
        # =========================
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )

        self.bn3 = nn.BatchNorm2d(F2)

        # =========================
        # 4. Classifier
        # =========================
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(F2, n_classes)
        )

    def forward(self, x):
        # x: (B, 1, C, T)

        # -------- Temporal --------
        x = self.temporal(x)
        x = self.bn1(x)
        x = F.elu(x)

        # x: (B, F1, C, T+1)

        # -------- Spatial --------
        x = self.spatial(x)
        x = self.bn2(x)
        x = F.elu(x)

        # x: (B, F1*D, 1, T+1)

        # -------- Separable --------
        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)

        # x: (B, F2, 1, T+2)

        # -------- Pooling --------
        x = torch.mean(x, dim=-1)

        # x: (B, F2, 1)

        x = x.squeeze(-1)

        # x: (B, F2)

        # -------- Classifier --------
        x = self.classifier(x)

        # x: (B, n_class)

        return x

if __name__ == '__main__':

    # test the model
    data = torch.rand(config.batch_size, 1, n_channels, n_times, device=config.device)
    model = EEGNetModel()
    model.to(config.device)
    logits = model(data)
    print(logits.shape)
