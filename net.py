"""Small custom net. conv1 mirrors AlexNet geometry: 448 -> 111 spatial grid.

conv1 is Gabor-initialised rather than random: an untrained random first layer
gives structurally meaningless edge response, and a Gabor bank is what conv1
converges to anyway. Deterministic, offline, no download.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device():
    """cuda > mps > cpu, overridable with BADAPPLE_DEVICE.

    The rest of the code takes `device` as an argument and is device-agnostic;
    this is the only place that chooses. `torch.backends.mps` exists on every
    platform since 1.12, but guard it anyway for older//stripped builds.
    """
    forced = os.environ.get("BADAPPLE_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def gabor_bank(n_filters=64, k=11, n_dc=16):
    """Gabor bank (edges) + a DC-sensitive low-pass set (fill).

    Zero-mean filters alone cannot distinguish a flat white region from a flat
    black one -- both give zero response -- so the silhouette interior would be
    unrecoverable. The low-pass filters carry brightness; paired signs let ReLU
    pass bright regions on one filter and dark regions on the other.
    """
    filts = []
    half = (k - 1) / 2
    y, x = torch.meshgrid(torch.arange(k) - half, torch.arange(k) - half, indexing="ij")
    n_or = 8
    for scale, sigma in enumerate([2.0, 3.4]):
        for o in range(n_or):
            th = math.pi * o / n_or
            xr = x * math.cos(th) + y * math.sin(th)
            yr = -x * math.sin(th) + y * math.cos(th)
            for lam in [4.0, 7.0]:
                for phase in [0.0, math.pi / 2]:
                    env = torch.exp(-(xr ** 2 + 0.7 * yr ** 2) / (2 * sigma ** 2))
                    g = env * torch.cos(2 * math.pi * xr / lam + phase)
                    g = g - g.mean()
                    g = g / (g.norm() + 1e-8)
                    filts.append(g)
    w = torch.stack(filts)[: max(0, n_filters - n_dc)]

    # DC / low-pass set: NOT mean-subtracted, so these encode brightness level.
    r2 = x ** 2 + y ** 2
    dc = []
    for i in range(max(0, n_dc)):
        sig = 1.6 + 0.9 * (i // 2)
        g = torch.exp(-r2 / (2 * sig ** 2))
        g = g / g.norm()
        dc.append(g if i % 2 == 0 else -g)           # paired signs: bright / dark
    if dc:
        w = torch.cat([w, torch.stack(dc)])
    w = w[:n_filters]
    return w.unsqueeze(1)                            # (n,1,k,k)


class MiniAlex(nn.Module):
    """448x448x1 -> conv1 111x111x64 -> mini-AlexNet head -> num_classes logits.

    num_classes is 4 everywhere: the four rotations of the self-supervised
    pretext task in train.py (0/90/180/270 degrees).

    The head deliberately keeps spatial extent through conv2/conv3 before pooling.
    A global pool applied directly to conv1 would make dL/da spatially uniform
    and flatten the landscape entirely.
    """

    def __init__(self, num_classes=4, width=64, logit_scale=4.0, gabor=True,
                 n_dc=16, pool="avg", seed=0):
        super().__init__()
        torch.manual_seed(seed)
        # MaxPool routes gradient only to winning positions, leaving ~2/3 of the
        # grid at exactly zero -- the surface renders as static rather than
        # terrain. AvgPool spreads gradient densely and is the better default.
        # It must be NON-overlapping: AvgPool2d(3, 2) gives overlapping windows
        # whose gradient coverage is non-uniform, stamping a checkerboard lattice
        # across the heightmap that shimmers under animation.
        POOLS = {"avg":  lambda: nn.AvgPool2d(2, 2),     # correct: dense, no lattice
                 "avg3": lambda: nn.AvgPool2d(3, 2),     # overlapping -> checkerboard
                 "max":  lambda: nn.MaxPool2d(3, 2)}     # sparse -> static
        P = POOLS[pool]
        self.conv1 = nn.Conv2d(1, width, kernel_size=11, stride=4, padding=2)
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            P(),
            nn.Conv2d(width, 192, 5, padding=2), nn.ReLU(inplace=True),
            P(),
            nn.Conv2d(192, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(3), nn.Flatten(),
            nn.Linear(256 * 9, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        if gabor:
            with torch.no_grad():
                self.conv1.weight.copy_(gabor_bank(width, 11, n_dc))
                self.conv1.bias.zero_()
        self.logit_scale = logit_scale

    def pre_relu(self, x):
        """conv1 output *before* ReLU -- this is `a`, the first hidden layer.

        Input is mapped [0,1] -> [-1,1] so that black is a signal, not an
        absence: with x=0 every filter responds zero regardless of its weights.
        """
        return self.conv1(2.0 * x - 1.0)

    def from_pre_relu(self, a):
        return self.head(a) * self.logit_scale

    def forward(self, x):
        return self.from_pre_relu(self.pre_relu(x))


def load_alexnet_conv1(model):
    """Optional: transplant real pretrained AlexNet conv1 (RGB weights summed to 1ch)."""
    from torchvision.models import alexnet, AlexNet_Weights
    w = alexnet(weights=AlexNet_Weights.DEFAULT).features[0].weight.data
    with torch.no_grad():
        model.conv1.weight.copy_(w.sum(1, keepdim=True)[: model.conv1.out_channels])
    return model
