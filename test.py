import torch
import torch.nn as nn

i, j = torch.meshgrid(
    torch.arange(6, dtype=torch.float32),
    torch.arange(6, dtype=torch.float32),
    indexing='xy'
)
print(i)
print(j)
print(i.shape, j.shape)

directions = torch.stack([(i-0)/1,
                          -(j-0)/1,
                          -torch.ones_like(i)], dim=-1)
print(directions.shape)
print(directions)
print(directions[..., None, :].shape)