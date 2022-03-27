#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
from torch import nn
from src.util import no_op

# Model initialization functions


def original_init_(model, extrema: Tuple[float, float] = (-0.3, 0.3)) -> None:
    for name, param in model.named_parameters():
        if name.endswith("weight"):
            nn.init.uniform_(param, a=extrema[0], b=extrema[1])
        if name.endswith("bias"):
            nn.init.zeros_(param)


def modern_init_(model) -> None:
    # Already the default in PyTorch
    # I.e.: Weights -> Kaiming | Bias -> Uniform with weight-dependent extrema
    # See: https://github.com/pytorch/pytorch/blob/7c2103ad5ffdc1ef91231c966988f7f2a61b4166/torch/nn/modules/linear.py#L92
    model.reset_parameters()


if __name__ == "__main__":
    no_op()
