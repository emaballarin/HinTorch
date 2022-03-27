#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sequence, List, Tuple
from itertools import product
import torch as th
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from src.util import no_op


# The algorithmic function (symmetry-detection for 6-sized binary inputs) we
# want to approximate with a NN
def is_symmetric(sequence: Sequence) -> float:
    assert len(sequence) == 6
    if (
        # sequence[0:3] == sequence[5:2:-1] still unsupported for PyTorch tensors
        sequence[0] == sequence[-1]
        and sequence[1] == sequence[-2]
        and sequence[2] == sequence[-3]
    ):
        # 1 == Yes | 0 == No
        return 1.0
    return 0.0


# The raw dataset...
x_all: List[Tuple[float]] = list(product([0.0, 1.0], repeat=6))
x_symmetric: List[Tuple[float]] = [item for item in x_all if is_symmetric(item)]
x_non_symmetric: List[Tuple[float]] = list(set(x_all).difference(set(x_symmetric)))

# ...tensorized
del x_all
x_symmetric: Tensor = th.tensor(x_symmetric, dtype=th.float32)
x_non_symmetric: Tensor = th.tensor(x_non_symmetric, dtype=th.float32)

# The unbalanced dataset tensor...
x: Tensor = th.cat((x_non_symmetric, x_symmetric), dim=0)
y: Tensor = th.tensor([[is_symmetric(sub_x)] for sub_x in x])

# ...and the balanced one
balancing_ratio: int = int((x_non_symmetric.shape[0] / x_symmetric.shape[0]))
x_balanced: Tensor = th.cat(
    (x_non_symmetric, th.cat([x_symmetric] * balancing_ratio, dim=0)), dim=0
)
y_balanced: Tensor = th.tensor([[is_symmetric(sub_x)] for sub_x in x_balanced])
del balancing_ratio

# Derived datasets...
train_unbalanced_ds: TensorDataset = TensorDataset(x, y)
train_balanced_ds: TensorDataset = TensorDataset(x_balanced, y_balanced)

# ...and Dataloaders (we do full-dataset-batching as in the paper)
train_unbalanced_dl: DataLoader = DataLoader(
    train_unbalanced_ds, batch_size=len(train_unbalanced_ds), shuffle=True
)
train_balanced_dl: DataLoader = DataLoader(
    train_balanced_ds, batch_size=len(train_balanced_ds), shuffle=True
)


if __name__ == "__main__":
    no_op()
