import torch
import torch.nn as nn

from typing import Tuple
from overrides import overrides


class RNNDropout(nn.Module):

    def __init__(
        self,
        p: float=0.5
    ):
        super().__init__()
        self.p = p
        
    @overrides
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if not self.training or self.p == 0.:
            return x
        shape = (x.size(0), 1, x.size(2))
        m = self.dropout_mask(x.data, shape, self.p)
        return x * m
        
    @staticmethod
    def dropout_mask(
        x: torch.Tensor,
        sz: Tuple[int, ...],
        p: float,
    ) -> torch.Tensor:
        return x.new(*sz).bernoulli_(1-p).float().div_(1-p)
