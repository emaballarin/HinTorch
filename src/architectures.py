#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from torch import nn
from ebtorch.nn import FCBlock
from src.util import no_op

# Model constants
MODEL_IN_SIZES: List[int] = [6, 2]
MODEL_OUT_SIZE: int = 1
MODEL_BIAS: bool = True
MODEL_DROPOUT: bool = False
MODEL_BATCHNORM: bool = False

# Model definition

model_original = FCBlock(
    in_sizes=MODEL_IN_SIZES,
    out_size=MODEL_OUT_SIZE,
    bias=MODEL_BIAS,
    activation_fx=nn.Sigmoid(),
    dropout=MODEL_DROPOUT,
    batchnorm=MODEL_BATCHNORM,
)

model_improved = FCBlock(
    in_sizes=MODEL_IN_SIZES,
    out_size=MODEL_OUT_SIZE,
    bias=MODEL_BIAS,
    activation_fx=[nn.Tanh(), nn.Sigmoid()],
    dropout=MODEL_DROPOUT,
    batchnorm=MODEL_BATCHNORM,
)

model_relu = FCBlock(
    in_sizes=MODEL_IN_SIZES,
    out_size=MODEL_OUT_SIZE,
    bias=MODEL_BIAS,
    activation_fx=[nn.ReLU(), nn.Sigmoid()],
    dropout=MODEL_DROPOUT,
    batchnorm=MODEL_BATCHNORM,
)

model_mish = FCBlock(
    in_sizes=MODEL_IN_SIZES,
    out_size=MODEL_OUT_SIZE,
    bias=MODEL_BIAS,
    activation_fx=[nn.Mish(), nn.Sigmoid()],
    dropout=MODEL_DROPOUT,
    batchnorm=MODEL_BATCHNORM,
)


if __name__ == "__main__":
    no_op()
