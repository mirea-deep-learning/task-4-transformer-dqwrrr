import torch
from torch import nn
from einops import rearrange, reduce

class SEBlock(nn.Module):
    """
    Implementation of the Squeeze-and-Excitation (SE) block proposed in [1].
    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    reduction : int, optional, default=16
        Reduction ratio to control the intermediate channel dimension.
    References
    ----------
    1. "`Squeeze-and-Excitation Networks. <https://arxiv.org/abs/1709.01507>`_" Jie Hu, et al. CVPR 2018.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16
    ) -> None:
        super(SEBlock, self).__init__()
        out_channels = max(1, in_channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, in_channels, height, width)
            Input tensor.
        Returns
        -------
        out : torch.Tensor (batch_size, in_channels, height, width)
            Output of the SK convolution layer.
        """
        # x: [b, c, h, w]
        # eq.2: Global average pooling (squeeze)
        z = reduce(x, 'b c h w -> b c', 'mean')  # [b, c]
        
        # eq.3: Excitation
        s = self.excitation(z)  # [b, c]
        s = rearrange(s, 'b c -> b c 1 1')  # [b, c, 1, 1]
        
        # eq.4: Scale input features
        out = x * s  # [b, c, h, w]
        return out