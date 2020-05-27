import torch
import torch.nn as nn

from typing import Union, Tuple
from overrides import overrides

from linear import Linear
from rnndropout import RNNDropout


class HighwayBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        first: str=False,
        couple: str=False,
        dropout_p: float=0.0,
        init_weight: float='kaiming',
        init_bias: Union[int, float, str]=-1
    ):
        super().__init__()
        self.first = first
        self.couple = couple
        if first:
            self.W_H = Linear(in_features, out_features, bias=False, activation=None)
            self.W_T = Linear(in_features, out_features, bias=False, activation=None)
            if not couple:
                self.W_C = Linear(in_features, out_features, bias=False, activation=None)
        self.R_H = Linear(in_features, out_features, bias=True, activation=None)
        self.R_T = Linear(in_features, out_features, bias=True, activation=None)
        if not couple:
            self.R_C = Linear(in_features, out_features, bias=True, activation=None)
        for child in self.children():
            child.reset_parameters(init_weight, init_bias)
        self.dropout = RNNDropout(dropout_p)

    @overrides
    def forward(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor
    ) -> torch.Tensor:
        if self.first:
            h = torch.tanh(self.W_H(x) + self.R_H(s))
            t = torch.sigmoid(self.W_T(x) + self.R_T(s))
            if self.couple:
                c = 1 - t
            else:
                c = torch.sigmoid(self.W_C(x) + self.R_C(s))
        else:
            h = torch.tanh(self.R_H(s))
            t = torch.sigmoid(self.R_T(s))
            if self.couple:
                c = 1 - t
            else:
                c = torch.sigmoid(self.R_C(s))
        t = self.dropout(t.unsqueeze(0)).squeeze(0)
        return h * t + s * c


class RecurrentHighwayNetwork(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        recurrence_depth: int=5,
        couple: str=False,
        dropout_p: float=0.,
        init_weight: str='kaiming',
        init_bias: Union[int, float, str]=-1
    ):
        super().__init__()
        self.highways = nn.ModuleList(
            [
                HighwayBlock(
                    in_features if l == 0 else out_features, 
                    out_features,
                    first=True if l == 0 else False,
                    couple=couple,
                    dropout_p=dropout_p,
                    init_weight=init_weight,
                    init_bias=init_bias
                )
                for l in range(recurrence_depth)
            ]
        )
        self.recurrence_depth = recurrence_depth
        # self.hidden_dim = out_features

    @overrides
    def forward(
        self, 
        input: torch.Tensor, 
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expects input shape: (seq_len, bsz, input_dim)
        outputs = []
        for x in input:
            for block in self.highways:
                hidden = block(x, hidden)
                outputs.append(hidden)
        outputs = torch.stack(outputs)
        return outputs, hidden
