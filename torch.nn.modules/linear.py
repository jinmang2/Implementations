import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union
from overrides import overrides


possible_activations = [
    'sigmoid', 'tanh', 'relu', 'rrelu', 'elu', 'hardtanh',
    'celu', 'selu', 'glu', 'gelu', 'hardshrink', 'leaky_relu'
    'logsigmoid', 'softplus', 'softshrink', 'prelu', 'softsign',
    'tanhshrink', 'softmin', 'softmax', 'log_softmax',
]


class MyLinear(nn.Module):

    __activations__ = activations
    __constants__ = [
        'bias', 'in_features', 'out_features', 
        'activation', 'use_functional'
    ]

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        bias: bool=True, 
        activation: Optional[str]=None,
        use_functional: bool=True,
    ):
        super().__init__()
        self.use_F = use_functional
        self.in_features = input_dim
        self.out_features = output_dim
        self.activation = activation
        if self.activation is not None:
            if activation.lower() in self.__activations__:
                setattr(self, 'activation', getattr(F, activation))
            else:
                raise Exception(f"Unknown activation function: {activation}")
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            # self.register_backward_hook('bias', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(
        self,
        init_weight: str='he', 
        init_bias: str='uniform',
        a: Union[int, float]=math.sqrt(5),
        gain: Union[int, float]=1.
    ):
        if init_weight.lower() in ['xavier', 'glorot']:
            nn.init.kaiming_uniform(self.weight, a=math.sqrt(5))
        elif init_weight.lower() in ['kaiming', 'he']:
            nn.init.xavier_uniform_(self.weight, gain=1.)
        else:
            raise ValueError(f"Unknown init_weight: {init_weight}")
        self.init_weight = init_weight
        if self.bias is not None:
            if init_bias == 'uniform':
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            elif init_bias in ['zero', 'zeros']:
                nn.init.zeros_(self.bias)
            elif isinstance(init_bias, int):
                self.bias.data.fill_(init_bias)

    @overrides
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_F:
            return F.linear(input, self.weight, self.bias)
        # expected input shape is (b, m, n)
        # since weight shape is (m, n), use matmul!
        output = input.matmul(self.weight.T)
        if self.bias is not None:
            output += self.bias
        if self.activation:
            output = self.activation(output)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
