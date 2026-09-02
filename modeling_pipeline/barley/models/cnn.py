"""ResNet-18 with a spectral stem, for 192-band masked kernel patches.

TWO CHANGES TO THE STOCK NETWORK
1. `conv1` takes 192 channels instead of 3. There is no useful ImageNet
   initialisation for that, so the stem is trained from scratch. A PCA stem is
   offered as an alternative (see `stem="pca3"`): project 192 bands onto 3
   components and use the pretrained network unchanged. At 548 kernels that
   often beats a from-scratch 192-channel stem, and it costs one flag to try.
2. The initial maxpool is dropped. Patches are 128x64, and the stock stem
   (stride-2 conv then stride-2 pool) would reach a 2x1 final feature map;
   without the pool it is 4x2, which leaves the network something to pool over.

EXPECTATIONS
548 kernels across 25 dishes is a small dataset for 11M parameters. PLS is a
serious baseline here, not a formality. If this does not beat it, that is a
result about the dataset size rather than a tuning failure -- and the number
that decides it is the dish-grouped one.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18


class SpectralResNet18(nn.Module):
    def __init__(self, n_channels, n_classes, stem="direct", pretrained=False,
                 drop_maxpool=True, dropout=0.0):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        net = resnet18(weights=weights)
        self.stem_kind = stem

        if stem == "direct":
            net.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2,
                                  padding=3, bias=False)
            self.reduce = None
        elif stem == "pca3":
            # A learned 1x1 projection to 3 channels, so the pretrained conv1
            # stays meaningful. Initialised from PCA when fit_reduce is called.
            self.reduce = nn.Conv2d(n_channels, 3, kernel_size=1, bias=False)
        else:
            raise ValueError(f"unknown stem {stem!r}")

        if drop_maxpool:
            net.maxpool = nn.Identity()
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, n_classes)) \
            if dropout else nn.Linear(512, n_classes)
        self.net = net

    @torch.no_grad()
    def fit_reduce(self, X):
        """Initialise the pca3 stem from the data's principal components.

        X is (N, C) spectra. Only meaningful for stem="pca3".
        """
        if self.reduce is None:
            return
        X = np.asarray(X, np.float64)
        X = X - X.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        W = torch.tensor(Vt[:3], dtype=torch.float32)
        self.reduce.weight.copy_(W.view(3, -1, 1, 1))

    def forward(self, x):
        if self.reduce is not None:
            x = self.reduce(x)
        return self.net(x)


def build(n_channels, n_classes, **kw):
    return SpectralResNet18(n_channels, n_classes, **kw)
