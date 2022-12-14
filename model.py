import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Callable


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        frequency: int,
    ):
        super().__init__()
        self.frequency = frequency
        self.embedding_functions = []

        frequency_range = 2**torch.linspace(0, self.frequency-1, self.frequency)

        for frequency in frequency_range:
            self.embedding_functions.append(lambda x: torch.sin(frequency*torch.pi*x))
            self.embedding_functions.append(lambda x: torch.cos(frequency*torch.pi*x))
        
    def forward(self, x) -> torch.Tensor:
        return torch.concat([embedding_function(x) for embedding_function in self.embedding_functions], dim=-1)


class Nerf(nn.Module):
    def __init__(self, input_dimension: int = 3, number_of_layers: int = 8, filter_dimension: int = 256, skip: List[int] = [4], view_direction_dimension: Optional[int] = None):
        super().__init__()
        self.input_dimension = input_dimension
        self.skip = skip
        assert (i < number_of_layers for i in self.skip)
        self.activation_function = nn.functional.relu
        self.view_direction_dimension = view_direction_dimension

        self.layers = nn.ModuleList(
            [nn.Linear(self.input_dimension, filter_dimension)] + 
            [nn.Linear(filter_dimension + self.input_dimension, filter_dimension) if i in skip else nn.Linear(filter_dimension, filter_dimension) for i in range(number_of_layers - 1)]
        )

        if self.view_direction_dimension is not None:
            self.output_sigma = nn.Linear(filter_dimension, 1)
            self.rgb_filters = nn.Linear(filter_dimension, filter_dimension)
            self.branch = nn.Linear(filter_dimension + self.view_direction_dimension, filter_dimension//2)
            self.output = nn.Linear(filter_dimension // 2, 3)
        else:
            self.output = nn.Linear(filter_dimension, 4)

    def forward(self, x: torch.Tensor, view_directions: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_x = x
        for i, layer in enumerate(self.layers):
            x = self.activation_function(layer(x))
            if i in self.skip:
                x = torch.cat([x, input_x], dim=-1)

        if self.view_direction_dimension is not None and view_directions is not None:
            sigma = self.output_sigma(x)

            x = self.rgb_filters(x)
            x = torch.concat([x, view_directions], dim=-1)
            x = self.activation_function(self.branch(x))
            x = self.output(x) # RGB

            x = torch.concat([x, sigma], dim=-1)
        else:
            x = self.output(x)

        return x
