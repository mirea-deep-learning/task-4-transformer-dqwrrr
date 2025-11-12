import torch
from torch import nn
from typing import List, Optional
from einops import rearrange, reduce

class SKConv(nn.Module):
    """
    Implementation of the Selective Kernel (SK) Convolution proposed in [1].
    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernels : List[int], optional, default=[3, 5]
        List of kernel sizes for each branch.
    reduction : int, optional, default=16
        Reduction ratio to control the dimension of "compact feature" ``z`` (see eq.4).
    L : int, optional, default=32
        Minimal value of the dimension of "compact feature" ``z`` (see eq.4).
    groups : int, optional, default=32
        Hyperparameter for ``torch.nn.Conv2d``.
    References
    ----------
    1. "`Selective Kernel Networks. <https://arxiv.org/abs/1903.06586>`_" Xiang Li, et al. CVPR 2019.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernels: List[int] = [3, 5],
        reduction: int = 16,
        L: int = 32,
        groups: int = 32
    ) -> None:
        super(SKConv, self).__init__()

        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        d = max(L, out_channels // reduction) # eq.4

        self.M = len(kernels)

        self.convs = nn.ModuleList([
                nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, k, padding=k//2, groups=groups, bias=False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)
            )
            for k in kernels
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_z = nn.Sequential(
            nn.Linear(out_channels, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True))
        self.fc_attn = nn.Linear(d, out_channels * self.M)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            Input tensor.
        Returns
        -------
        out : torch.Tensor (batch_size, out_channels, height, width)
            Output of the SK convolution layer.
        """
        #Conv2d , AvgPoll, softmax, ReLU, BatchNorm, Linear

        # ----- split -----
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)  # [b, M, c, h, w]

        # ----- fuse -----
        # eq.1: Sum over M branches
        U = reduce(feats, 'b M c h w -> b c h w', 'sum')  # [b, c, h, w]
        
        # eq.2: Global average pooling
        s = reduce(U, 'b c h w -> b c', 'mean')  # [b, c]
        
        # eq.3: Compact feature
        z = self.fc_z(s)  # [b, d]

        # ----- select -----
        batch_size = s.shape[0]

        # eq.5: Attention map
        score = self.fc_attn(z)  # [b, M * c]
        att = rearrange(score, 'b (M c) -> b M c 1 1', M=self.M, c=self.out_channels)
        att = self.softmax(att)

        # eq.6: Fuse branches with attention weights
        out = reduce(feats * att, 'b M c h w -> b c h w', 'sum')  # [b, c, h, w]
        return out

print("=== Testing SKConv ===")
features = torch.rand(1, 34*16, 25, 25)
out = SKConv(34*16).eval()
result = out(features)
print(f"SKConv input shape: {features.shape}")
print(f"SKConv output shape: {result.shape}")
print()

print("=== Testing Regular Conv2d ===")
n = nn.Conv2d(3, 3, kernel_size=3)
print(f"Regular Conv2d weight shape: {n.weight.shape}")

print("\n=== Testing Grouped Conv2d ===")
n = nn.Conv2d(3, 3, kernel_size=3, groups=3)
print(f"Grouped Conv2d weight shape: {n.weight.shape}")

print("\n=== Testing Conv2d Forward Pass ===")
features = torch.rand(1, 3, 25, 25)
output = n(features)
print(f"Conv2d input shape: {features.shape}")
print(f"Conv2d output shape: {output.shape}")