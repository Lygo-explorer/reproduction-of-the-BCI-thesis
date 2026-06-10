from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def build_static_adjacency(
    distances: Tensor,
    dis_lower: float = 5.0,
    dis_upper: float = 40.0,
    w_min: float = 0.2,
    w_max: float = 1.0,
) -> Tensor:
    """Build the distance-thresholded static adjacency matrix from the paper."""
    if distances.ndim != 2 or distances.size(0) != distances.size(1):
        raise ValueError("distances must be a square [num_channels, num_channels] matrix")

    d = distances.float()
    adj = w_min + (w_max - w_min) * (dis_upper - d) / (dis_upper - dis_lower)
    adj = torch.where(d <= dis_lower, torch.full_like(adj, w_max), adj)
    adj = torch.where(d >= dis_upper, torch.full_like(adj, w_min), adj)
    adj = adj.clamp(min=0.0, max=w_max)
    adj = adj.masked_fill(torch.eye(d.size(0), device=d.device, dtype=torch.bool), 0.0)
    return adj


class BandEnergy(nn.Module):
    """Windowed DFT band-energy transform for multichannel sEMG.

    Input shape is [batch, channels, frames]. Output shape is
    [batch, windows, channels, bands].
    """

    def __init__(
        self,
        window_size: int = 300,
        overlap: float = 0.6,
        num_bands: int = 8,
        band_edges: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1")
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")

        self.window_size = window_size
        self.overlap = overlap
        self.hop_size = max(1, int(round(window_size * (1.0 - overlap))))
        self.num_bands = num_bands
        self.band_edges = tuple(band_edges) if band_edges is not None else None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, channels, frames]")
        if x.size(-1) < self.window_size:
            raise ValueError("frames must be at least window_size")

        windows = x.unfold(dimension=-1, size=self.window_size, step=self.hop_size)
        windows = windows.transpose(1, 2).contiguous()
        spectrum = torch.fft.rfft(windows, dim=-1)
        power = spectrum.abs().square()
        return self._pool_bands(power)

    def _pool_bands(self, power: Tensor) -> Tensor:
        freq_bins = power.size(-1)
        if self.band_edges is None:
            edges = torch.linspace(
                0,
                freq_bins,
                self.num_bands + 1,
                device=power.device,
            ).round().long()
        else:
            edges = torch.tensor(self.band_edges, device=power.device, dtype=torch.long)
            if edges.numel() != self.num_bands + 1:
                raise ValueError("band_edges length must equal num_bands + 1")

        bands = []
        for start, end in zip(edges[:-1].tolist(), edges[1:].tolist()):
            end = max(end, start + 1)
            bands.append(power[..., start:end].mean(dim=-1))
        return torch.stack(bands, dim=-1)


class StaticSourceModel(nn.Module):
    """Static sEMG source task classifier used for knowledge transfer.

    The paper uses DFT band energy followed by a three-layer fully connected
    network. This module can receive raw windows [B, C, T] or precomputed
    features [B, C, bands].
    """

    def __init__(
        self,
        num_channels: int,
        num_bands: int = 8,
        num_static_classes: int = 5,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        window_size: int = 300,
        overlap: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.band_energy = BandEnergy(window_size, overlap, num_bands)

        dims = [num_channels * num_bands, *hidden_dims, num_static_classes]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3 and x.size(-1) != self.num_bands:
            features = self.band_energy(x).mean(dim=1)
        elif x.ndim == 3:
            features = x
        else:
            raise ValueError("x must have shape [B, C, T] or [B, C, bands]")
        return self.classifier(features.flatten(start_dim=1))

    def predict_proba(self, x: Tensor) -> Tensor:
        return F.softmax(self.forward(x), dim=-1)


class TemporalTransferModule(nn.Module):
    """Static-to-sequential transfer module.

    Each 300-frame sequential atom is classified by the frozen or trainable
    static source model; the resulting probability sequence is flattened and
    projected into a temporal feature vector.
    """

    def __init__(
        self,
        source_model: StaticSourceModel,
        max_atoms: int,
        temporal_dim: int = 128,
        hidden_dim: int = 256,
        freeze_source: bool = True,
    ) -> None:
        super().__init__()
        self.source_model = source_model
        self.max_atoms = max_atoms
        self.num_static_classes = source_model.classifier[-1].out_features
        if freeze_source:
            for param in self.source_model.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(max_atoms * self.num_static_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, temporal_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        atoms = self._extract_atoms(x)
        batch, num_atoms, channels, frames = atoms.shape
        flat_atoms = atoms.reshape(batch * num_atoms, channels, frames)

        with torch.set_grad_enabled(any(p.requires_grad for p in self.source_model.parameters())):
            probs = self.source_model.predict_proba(flat_atoms)
        probs = probs.reshape(batch, num_atoms, self.num_static_classes)
        padded = self._fit_atom_count(probs)
        return self.projector(padded.flatten(start_dim=1)), probs

    def _extract_atoms(self, x: Tensor) -> Tensor:
        band_energy = self.source_model.band_energy
        windows = x.unfold(-1, band_energy.window_size, band_energy.hop_size)
        return windows.transpose(1, 2).contiguous()

    def _fit_atom_count(self, probs: Tensor) -> Tensor:
        if probs.size(1) == self.max_atoms:
            return probs
        if probs.size(1) > self.max_atoms:
            return probs[:, : self.max_atoms]

        pad = probs.new_zeros(probs.size(0), self.max_atoms - probs.size(1), probs.size(2))
        return torch.cat([probs, pad], dim=1)


class DynamicGraphConv(nn.Module):
    """Weighted graph convolution with per-sample adjacency."""

    def __init__(self, in_features: int, out_features: int, add_self_loops: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.add_self_loops = add_self_loops

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        if adjacency.ndim == 2:
            adjacency = adjacency.expand(x.size(0), x.size(1), -1, -1)
        if adjacency.ndim != 4:
            raise ValueError("adjacency must be [C, C] or [B, W, C, C]")

        adjacency = adjacency.to(device=x.device, dtype=x.dtype)
        if self.add_self_loops:
            eye = torch.eye(adjacency.size(-1), device=x.device, dtype=x.dtype)
            adjacency = adjacency + eye

        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        norm_adj = adjacency / degree
        aggregated = torch.einsum("bwic,bwcf->bwif", norm_adj, x)
        return F.relu(self.linear(aggregated))


class SpatialFeatureModule(nn.Module):
    """Adaptive dynamic/static GCN branch from STFEN."""

    def __init__(
        self,
        num_channels: int,
        static_adjacency: Tensor,
        num_bands: int = 8,
        gcn_hidden_dim: int = 32,
        spatial_dim: int = 128,
        window_size: int = 300,
        overlap: float = 0.6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.band_energy = BandEnergy(window_size, overlap, num_bands)
        self.register_buffer("static_adjacency", static_adjacency.float())

        self.static_gcn1 = DynamicGraphConv(num_bands, gcn_hidden_dim)
        self.dynamic_gcn1 = DynamicGraphConv(num_bands, gcn_hidden_dim)
        self.static_gcn2 = DynamicGraphConv(gcn_hidden_dim, gcn_hidden_dim)
        self.dynamic_gcn2 = DynamicGraphConv(gcn_hidden_dim, gcn_hidden_dim)

        pooled_dim = gcn_hidden_dim * 4 * 2
        self.projector = nn.Sequential(
            nn.Linear(pooled_dim, spatial_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim, spatial_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        features = self.band_energy(x)
        dynamic_adj1 = cosine_adjacency(features)
        static_adj = self.static_adjacency

        h_static1 = self.static_gcn1(features, static_adj)
        h_dynamic1 = self.dynamic_gcn1(features, dynamic_adj1)

        dynamic_adj2 = cosine_adjacency(h_static1)
        h_static2 = self.static_gcn2(h_static1, static_adj)
        h_dynamic2 = self.dynamic_gcn2(h_static1, dynamic_adj2)

        graph_features = torch.cat([h_static1, h_dynamic1, h_static2, h_dynamic2], dim=-1)
        max_pool = graph_features.amax(dim=(1, 2))
        avg_pool = graph_features.mean(dim=(1, 2))
        spatial = self.projector(torch.cat([max_pool, avg_pool], dim=-1))

        intermediates = {
            "band_energy": features,
            "dynamic_adjacency_layer1": dynamic_adj1,
            "dynamic_adjacency_layer2": dynamic_adj2,
            "graph_features": graph_features,
        }
        return spatial, intermediates


def cosine_adjacency(features: Tensor) -> Tensor:
    """Cosine-similarity adjacency over channels for [B, W, C, F] tensors."""
    normalized = F.normalize(features, p=2, dim=-1, eps=1e-6)
    adjacency = torch.einsum("bwif,bwjf->bwij", normalized, normalized).clamp_min(0.0)
    eye = torch.eye(adjacency.size(-1), device=adjacency.device, dtype=torch.bool)
    return adjacency.masked_fill(eye, 0.0)


@dataclass(frozen=True)
class STFENOutput:
    logits: Tensor
    temporal_features: Tensor
    spatial_features: Tensor
    static_probabilities: Tensor
    spatial_intermediates: dict[str, Tensor]


class STFEN(nn.Module):
    """Spatio-Temporal Feature Extraction Network for sequential sEMG."""

    def __init__(
        self,
        num_channels: int,
        num_sequence_classes: int,
        static_adjacency: Tensor,
        sequence_frames: int,
        num_bands: int = 8,
        num_static_classes: int = 5,
        temporal_dim: int = 128,
        spatial_dim: int = 128,
        classifier_hidden_dim: int = 128,
        source_model: Optional[StaticSourceModel] = None,
        freeze_source: bool = True,
        window_size: int = 300,
        overlap: float = 0.6,
    ) -> None:
        super().__init__()
        if source_model is None:
            source_model = StaticSourceModel(
                num_channels=num_channels,
                num_bands=num_bands,
                num_static_classes=num_static_classes,
                window_size=window_size,
                overlap=overlap,
            )

        hop_size = max(1, int(round(window_size * (1.0 - overlap))))
        max_atoms = 1 + max(0, (sequence_frames - window_size) // hop_size)

        self.temporal = TemporalTransferModule(
            source_model=source_model,
            max_atoms=max_atoms,
            temporal_dim=temporal_dim,
            freeze_source=freeze_source,
        )
        self.spatial = SpatialFeatureModule(
            num_channels=num_channels,
            static_adjacency=static_adjacency,
            num_bands=num_bands,
            spatial_dim=spatial_dim,
            window_size=window_size,
            overlap=overlap,
        )
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim + spatial_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(classifier_hidden_dim, num_sequence_classes),
        )

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor | STFENOutput:
        temporal_features, static_probs = self.temporal(x)
        spatial_features, spatial_intermediates = self.spatial(x)
        logits = self.classifier(torch.cat([temporal_features, spatial_features], dim=-1))

        if not return_features:
            return logits
        return STFENOutput(
            logits=logits,
            temporal_features=temporal_features,
            spatial_features=spatial_features,
            static_probabilities=static_probs,
            spatial_intermediates=spatial_intermediates,
        )
