# Import Libraries
import torch
import torch.nn as nn
import math



class NewGELU(nn.Module):
    """
    Reference: Gaussian Error Linear Units (GELU) paper:
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )