import torch
import torch.nn.functional as F
from typing import Optional


class Softmax(nn.Module):

    def __init__(
        self,
        dim: Optional[int]=None,
        use_functional: bool=True,
    ):
        super().__init__()
        self.dim = dim
        self.use_F = use_functional

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    @overrides
    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        if self.use_F:
            return F.softmax(input, self.dim, _stacklevel=5)
        if self.dim is None:
            ndim = input.ndim
            if ndim in [0, 1, 3]:
                dim = 0
            else:
                dim = 1
        else:
            dim = self.dim
        input_max = input.max(dim=dim).values
        input_max = input_max.unsqueeze(dim=dim)
        exp_x_c = torch.exp(input - input_max)
        sum_exp_x_c = exp_x_c.sum(dim=dim).unsqueeze(dim=dim)
        return exp_x_c / sum_exp_x_c

    def extra_repr(self) -> str:
        return f'dim={self.dim}'
