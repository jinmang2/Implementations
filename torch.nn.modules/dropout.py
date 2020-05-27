import torch
import torch.nn as nn
import torch.distributions as torchdist


class Dropout(nn.Module):

    def __init__(
        self,
        p: float=0.5
    ):
        super().__init__()
        if p < 0. or p > 1.:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}.".format(p))
        self.p = p
    
    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        mask = torchdist.bernoulli.Bernoulli(1-self.p).sample(input.size())
        output = input*  mask / (1 - self.p)
        return output
        
    def extra_repr(self) -> str:
        return f"p={self.p}"
