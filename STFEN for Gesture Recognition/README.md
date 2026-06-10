# Sequential sEMG Recognition With Knowledge Transfer and Dynamic Graph Network Based on Spatio-Temporal Feature Extraction Network

This project implements the neural network structure from:

Sequential sEMG Recognition With Knowledge Transfer and Dynamic Graph Network
Based on Spatio-Temporal Feature Extraction Network, IEEE JBHI, 2025.

## Implemented Modules

- `BandEnergy`: 300-frame windowed DFT band-energy extraction.
- `StaticSourceModel`: static sEMG source classifier, implemented as DFT plus a three-layer MLP.
- `TemporalTransferModule`: static-to-sequential knowledge transfer. It converts sequential sEMG atoms into static action probability sequences and projects the flattened sequence into temporal features.
- `SpatialFeatureModule`: adaptive static/dynamic graph branch. It uses a static distance graph and cosine-similarity dynamic graph with two GCN layers, then combines max and average pooling.
- `STFEN`: full temporal/spatial fusion classifier.

## Input Shape

The full model expects raw sequential sEMG tensors shaped:

```text
[batch_size, num_channels, num_frames]
```

For the ADSE setting described in the paper, `num_channels=8` and the final classifier has `num_sequence_classes=10` for digits 0-9.

## Quick Demo

```powershell
python main.py
```

The demo creates random sEMG-like input and prints the shapes of logits, transferred static probabilities, and dynamic adjacency matrices.

## Notes

The paper describes the complete module logic but does not publish exact hidden layer widths. The implementation therefore exposes hidden dimensions as constructor parameters while keeping the paper's confirmed operations and data flow.
