import torch

from stfen import STFEN, build_static_adjacency


def main() -> None:
    batch_size = 2
    num_channels = 8
    frames = 1000
    num_digits = 10

    distances = torch.full((num_channels, num_channels), 20.0)
    distances.fill_diagonal_(0.0)
    static_adjacency = build_static_adjacency(distances)

    model = STFEN(
        num_channels=num_channels,
        num_sequence_classes=num_digits,
        static_adjacency=static_adjacency,
        sequence_frames=frames,
    )

    semg = torch.randn(batch_size, num_channels, frames)
    output = model(semg, return_features=True)
    print("logits:", tuple(output.logits.shape))
    print("static probability sequence:", tuple(output.static_probabilities.shape))
    print(
        "dynamic adjacency:",
        tuple(output.spatial_intermediates["dynamic_adjacency_layer2"].shape),
    )


if __name__ == "__main__":
    main()
