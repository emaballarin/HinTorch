# %% [markdown]
# # Is Deep Learning *slipping* [on the shoulders of Giants](https://www.wikiwand.com/en/Standing_on_the_shoulders_of_giants)?
# ### A *critical* reproducibility challenge for [[Rumelhart et al., 1986]](https://sci-hub.se/10.1038/323533a0)

# %% [markdown]
# *TO BE FILLED*
#
# Explain why and when it all started...

# %% [markdown]
# #### Imports

# %%
# Easily compute statistics of arrays
import numpy as np

# Type hints
from typing import Iterable, List, Tuple
from torch import Tensor

# Tensors and NNs
import torch as th
from ebtorch.nn import FCBlock  # API for fully-connected NN blocks
from torch import nn
import torch.nn.functional as F

# Optimizers
import torch.optim as optim

# Tensor data[sets|loader]s
from torch.utils.data import TensorDataset, DataLoader

# Iterable handling
from itertools import product
from copy import deepcopy

# Utilities for callables
from ebtorch.nn.utils import argser_f, emplace_kv

# %% [markdown]
# #### Datasets

# %%
# The algorithmic function (symmetry-detection for 6-sized binary inputs) we
# want to approximate with a NN
def is_symmetric(iterable: Iterable) -> float:
    assert len(iterable) == 6
    if (
        # iterable[0:3] == iterable[5:2:-1] still unsupported for PyTorch tensors
        iterable[0] == iterable[-1]
        and iterable[1] == iterable[-2]
        and iterable[2] == iterable[-3]
    ):
        # 1 == Yes | 0 == No
        return 1.0
    return 0.0


# %%
# We split the dataset output-wise early on, to be able to balance it later in
# case we need to.

x_all: List[Tuple[float]] = [item for item in product([0.0, 1.0], repeat=6)]
x_symmetric: List[Tuple[float]] = [item for item in x_all if is_symmetric(item)]
x_non_symmetric: List[Tuple[float]] = [
    item for item in set(x_all).difference(set(x_symmetric))
]

# And we tensorize it
del x_all
x_symmetric: Tensor = th.tensor(x_symmetric, dtype=th.float32)
x_non_symmetric: Tensor = th.tensor(x_non_symmetric, dtype=th.float32)

# %%
# The unbalanced dataset tensor
x: Tensor = th.cat((x_non_symmetric, x_symmetric), dim=0)
y: Tensor = th.tensor([[is_symmetric(sub_x)] for sub_x in x])

# And the balanced one
balancing_ratio: int = int((x_non_symmetric.shape[0] / x_symmetric.shape[0]))
x_balanced: Tensor = th.cat(
    (x_non_symmetric, th.cat([x_symmetric] * balancing_ratio, dim=0)), dim=0
)
y_balanced: Tensor = th.tensor([[is_symmetric(sub_x)] for sub_x in x_balanced])
del balancing_ratio

# %%
# Conversion to proper PyTorch data[set|loader]s

# Datasets
train_unbalanced_ds: TensorDataset = TensorDataset(x, y)
train_balanced_ds: TensorDataset = TensorDataset(x_balanced, y_balanced)

# Dataloaders (we do full-dataset-batching as in the paper)
train_unbalanced_dl: DataLoader = DataLoader(
    train_unbalanced_ds, batch_size=len(train_unbalanced_ds), shuffle=True
)
train_balanced_dl: DataLoader = DataLoader(
    train_balanced_ds, batch_size=len(train_balanced_ds), shuffle=True
)

# %% [markdown]
# #### Models

# %%
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

model_modern_init = FCBlock(
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

# %%
# Model initialization functions


def original_init_(model, extrema: Tuple[float]) -> None:
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


# %% [markdown]
# #### Training

# %%
# Run training until (hopeful) convergence; gather statistics


def train_diag_aio(
    model,
    dataloader: DataLoader,
    max_epochs_nr: int,
    loss,
    optimizer_fx,
    optimizer_dict: dict,
    device,
) -> Tuple[Tuple, Tuple[float]]:

    optimizer_params: dict = emplace_kv(optimizer_dict, "params", model.parameters())
    optimizer = argser_f(optimizer_fx, optimizer_params)()

    losses: list = []
    accuracies: List[float] = []

    # Move model to device
    model = model.to(device)

    # Put model in training mode
    model.train()

    # Iterate over epochs
    epoch: int
    for epoch in range(max_epochs_nr):

        # Iterate over batches
        # (in our case: batch == dataset)
        x: Tensor
        y: Tensor
        for x, y in dataloader:

            # Move batch to device
            x: Tensor
            y: Tensor
            x, y = x.to(device), y.to(device)

            # Forward pass
            optimizer.zero_grad()
            y_hat: Tensor = model(x)
            loss_value = loss(y_hat, y)

            # STATISTICS COMPUTATION
            # Here is fine, since batch == dataset
            with th.no_grad():
                pred = th.round(model(x))
                accuracy = (pred.eq(y.view_as(pred))).sum().item() / len(x)

            losses.append(loss_value.item())
            accuracies.append(accuracy)

            # Backward pass
            loss_value.backward()
            optimizer.step()

    model.eval()

    return tuple(deepcopy(losses)), tuple(deepcopy(accuracies))


# %% [markdown]
# #### Training hyperparameters

# %%
MAX_EPOCHS_NR: int = int(4e4)
LOSS = F.mse_loss
OPTIMIZER_FX = optim.SGD
OPTIMIZER_PARAMS: dict = {"params": None, "lr": 0.1, "momentum": 0.9}
DEVICE = "cpu"

SAMPLE_SIZE: int = 1

# Stuff
MAX_EPOCHS_NR: int = max(MAX_EPOCHS_NR, 1426)

# %% [markdown]
# ---

# %% [markdown]
# #### Training... **the original architecture**
#
# Overly simple, a bit naif, but the one that started it all (or not?)!

# %%
converged_epochs_list: List[int] = []
accuracies_list: List[float] = []
hintonpoint_acc_list: List[float] = []

# Iterate over realizations (samples) of training
sample_nr: int
for sample_nr in range(SAMPLE_SIZE):

    # Train model
    original_init_(model_original, extrema=(-0.3, 0.3))
    a: Tuple[float]
    _, a = train_diag_aio(
        model_original,
        train_unbalanced_dl,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
    )

    # Compute running stats
    accuracies_list.append(a[-1])
    hintonpoint_acc_list.append(a[1425])
    if a[-1] == 1.0:
        converged_epochs_list.append(a.index(1.0))

accuracies_np = np.array(accuracies_list)
hp_acc_list = np.array(hintonpoint_acc_list)
conv_epoch_np = np.array(converged_epochs_list)

print(" ")
print(
    f"AVERAGE ACCURACY AT {MAX_EPOCHS_NR} EPOCHS: {accuracies_np.mean()} (Std. Dev.: {accuracies_np.std()}) over {len(accuracies_np)} runs"
)
print(
    f"AVERAGE ACCURACY AT HINTON POINT ({int(1425)} EPOCHS): {hp_acc_list.mean()} (Std. Dev.: {hp_acc_list.std()}) over {len(hp_acc_list)} runs"
)
print(" ")
print(
    f"CONVERGED AT {MAX_EPOCHS_NR} EPOCHS: {len(conv_epoch_np)} over {len(accuracies_np)} runs"
)
print(
    f"CONVERGED AT HINTON POINT ({int(1425)} EPOCHS): {(hp_acc_list == 1.0).sum()} over {len(accuracies_np)} runs"
)
print(" ")
if len(conv_epoch_np) > 0:
    print(
        f"AVERAGE EPOCHS UNTIL CONVERGENCE: {conv_epoch_np.mean()} (Std. Dev.: {conv_epoch_np.std()})"
    )

# %% [markdown]
# #### Training... **the original architecture** on a **balanced dataset**
#
# Maybe they forgot to say...

# %%
converged_epochs_list: List[int] = []
accuracies_list: List[float] = []
hintonpoint_acc_list: List[float] = []

# Iterate over realizations (samples) of training
sample_nr: int
for sample_nr in range(SAMPLE_SIZE):

    # Train model
    original_init_(model_original, extrema=(-0.3, 0.3))
    a: Tuple[float]
    _, a = train_diag_aio(
        model_original,
        train_balanced_dl,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
    )

    # Compute running stats
    accuracies_list.append(a[-1])
    hintonpoint_acc_list.append(a[1425])
    if a[-1] == 1.0:
        converged_epochs_list.append(a.index(1.0))

accuracies_np = np.array(accuracies_list)
hp_acc_list = np.array(hintonpoint_acc_list)
conv_epoch_np = np.array(converged_epochs_list)

print(" ")
print(
    f"AVERAGE ACCURACY AT {MAX_EPOCHS_NR} EPOCHS: {accuracies_np.mean()} (Std. Dev.: {accuracies_np.std()}) over {len(accuracies_np)} runs"
)
print(
    f"AVERAGE ACCURACY AT HINTON POINT ({int(1425)} EPOCHS): {hp_acc_list.mean()} (Std. Dev.: {hp_acc_list.std()}) over {len(hp_acc_list)} runs"
)
print(" ")
print(
    f"CONVERGED AT {MAX_EPOCHS_NR} EPOCHS: {len(conv_epoch_np)} over {len(accuracies_np)} runs"
)
print(
    f"CONVERGED AT HINTON POINT ({int(1425)} EPOCHS): {(hp_acc_list == 1.0).sum()} over {len(accuracies_np)} runs"
)
print(" ")
if len(conv_epoch_np) > 0:
    print(
        f"AVERAGE EPOCHS UNTIL CONVERGENCE: {conv_epoch_np.mean()} (Std. Dev.: {conv_epoch_np.std()})"
    )

# %% [markdown]
# #### Training... **a slightly improved architecture**, with ***modern* initialization**, on a **balanced dataset**
#
# To be honest, it was considered the *standard* NN at the time (with the only exception of the initialization... but that's fine!)

# %%
converged_epochs_list: List[int] = []
accuracies_list: List[float] = []
hintonpoint_acc_list: List[float] = []

# Iterate over realizations (samples) of training
sample_nr: int
for sample_nr in range(SAMPLE_SIZE):

    # Train model
    modern_init_(model_improved)
    a: Tuple[float]
    _, a = train_diag_aio(
        model_improved,
        train_balanced_dl,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
    )

    # Compute running stats
    accuracies_list.append(a[-1])
    hintonpoint_acc_list.append(a[1425])
    if a[-1] == 1.0:
        converged_epochs_list.append(a.index(1.0))

accuracies_np = np.array(accuracies_list)
hp_acc_list = np.array(hintonpoint_acc_list)
conv_epoch_np = np.array(converged_epochs_list)

print(" ")
print(
    f"AVERAGE ACCURACY AT {MAX_EPOCHS_NR} EPOCHS: {accuracies_np.mean()} (Std. Dev.: {accuracies_np.std()}) over {len(accuracies_np)} runs"
)
print(
    f"AVERAGE ACCURACY AT HINTON POINT ({int(1425)} EPOCHS): {hp_acc_list.mean()} (Std. Dev.: {hp_acc_list.std()}) over {len(hp_acc_list)} runs"
)
print(" ")
print(
    f"CONVERGED AT {MAX_EPOCHS_NR} EPOCHS: {len(conv_epoch_np)} over {len(accuracies_np)} runs"
)
print(
    f"CONVERGED AT HINTON POINT ({int(1425)} EPOCHS): {(hp_acc_list == 1.0).sum()} over {len(accuracies_np)} runs"
)
print(" ")
if len(conv_epoch_np) > 0:
    print(
        f"AVERAGE EPOCHS UNTIL CONVERGENCE: {conv_epoch_np.mean()} (Std. Dev.: {conv_epoch_np.std()})"
    )

# %% [markdown]
# #### Training... **a ReLU Network**, with ***modern* initialization**, on a **balanced dataset**
#
# The default of defaults nowadays. It is the same thing Medium is filled up with: will it be up to any good?
# (or maybe... despite being a Medium trend 🙃)

# %%
converged_epochs_list: List[int] = []
accuracies_list: List[float] = []
hintonpoint_acc_list: List[float] = []

# Iterate over realizations (samples) of training
sample_nr: int
for sample_nr in range(SAMPLE_SIZE):

    # Train model
    modern_init_(model_relu)
    a: Tuple[float]
    _, a = train_diag_aio(
        model_relu,
        train_balanced_dl,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
    )

    # Compute running stats
    accuracies_list.append(a[-1])
    hintonpoint_acc_list.append(a[1425])
    if a[-1] == 1.0:
        converged_epochs_list.append(a.index(1.0))

accuracies_np = np.array(accuracies_list)
hp_acc_list = np.array(hintonpoint_acc_list)
conv_epoch_np = np.array(converged_epochs_list)

print(" ")
print(
    f"AVERAGE ACCURACY AT {MAX_EPOCHS_NR} EPOCHS: {accuracies_np.mean()} (Std. Dev.: {accuracies_np.std()}) over {len(accuracies_np)} runs"
)
print(
    f"AVERAGE ACCURACY AT HINTON POINT ({int(1425)} EPOCHS): {hp_acc_list.mean()} (Std. Dev.: {hp_acc_list.std()}) over {len(hp_acc_list)} runs"
)
print(" ")
print(
    f"CONVERGED AT {MAX_EPOCHS_NR} EPOCHS: {len(conv_epoch_np)} over {len(accuracies_np)} runs"
)
print(
    f"CONVERGED AT HINTON POINT ({int(1425)} EPOCHS): {(hp_acc_list == 1.0).sum()} over {len(accuracies_np)} runs"
)
print(" ")
if len(conv_epoch_np) > 0:
    print(
        f"AVERAGE EPOCHS UNTIL CONVERGENCE: {conv_epoch_np.mean()} (Std. Dev.: {conv_epoch_np.std()})"
    )

# %% [markdown]
# #### Training... **a Mish Network**, with ***modern* initialization**, on a **balanced dataset**
#
# Why would someone ever use an activation function that requires exponentiation, is non-monotonic, and looks like a neural action-potential?
#
# Hold my beer... 🍻

# %%
converged_epochs_list: List[int] = []
accuracies_list: List[float] = []
hintonpoint_acc_list: List[float] = []

# Iterate over realizations (samples) of training
sample_nr: int
for sample_nr in range(SAMPLE_SIZE):

    # Train model
    modern_init_(model_mish)
    a: Tuple[float]
    _, a = train_diag_aio(
        model_mish,
        train_balanced_dl,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
    )

    # Compute running stats
    accuracies_list.append(a[-1])
    hintonpoint_acc_list.append(a[1425])
    if a[-1] == 1.0:
        converged_epochs_list.append(a.index(1.0))

accuracies_np = np.array(accuracies_list)
hp_acc_list = np.array(hintonpoint_acc_list)
conv_epoch_np = np.array(converged_epochs_list)

print(" ")
print(
    f"AVERAGE ACCURACY AT {MAX_EPOCHS_NR} EPOCHS: {accuracies_np.mean()} (Std. Dev.: {accuracies_np.std()}) over {len(accuracies_np)} runs"
)
print(
    f"AVERAGE ACCURACY AT HINTON POINT ({int(1425)} EPOCHS): {hp_acc_list.mean()} (Std. Dev.: {hp_acc_list.std()}) over {len(hp_acc_list)} runs"
)
print(" ")
print(
    f"CONVERGED AT {MAX_EPOCHS_NR} EPOCHS: {len(conv_epoch_np)} over {len(accuracies_np)} runs"
)
print(
    f"CONVERGED AT HINTON POINT ({int(1425)} EPOCHS): {(hp_acc_list == 1.0).sum()} over {len(accuracies_np)} runs"
)
print(" ")
if len(conv_epoch_np) > 0:
    print(
        f"AVERAGE EPOCHS UNTIL CONVERGENCE: {conv_epoch_np.mean()} (Std. Dev.: {conv_epoch_np.std()})"
    )

# %% [markdown]
# #### Training... **a Mish Network**, with ***modern* initialization**, on the **unbalanced dataset**
#
# Who said balancing was necessary?
#
# (drops the beer)

# %%
converged_epochs_list: List[int] = []
accuracies_list: List[float] = []
hintonpoint_acc_list: List[float] = []

# Iterate over realizations (samples) of training
sample_nr: int
for sample_nr in range(SAMPLE_SIZE):

    # Train model
    modern_init_(model_mish)
    a: Tuple[float]
    _, a = train_diag_aio(
        model_mish,
        train_unbalanced_dl,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
    )

    # Compute running stats
    accuracies_list.append(a[-1])
    hintonpoint_acc_list.append(a[1425])
    if a[-1] == 1.0:
        converged_epochs_list.append(a.index(1.0))

accuracies_np = np.array(accuracies_list)
hp_acc_list = np.array(hintonpoint_acc_list)
conv_epoch_np = np.array(converged_epochs_list)

print(" ")
print(
    f"AVERAGE ACCURACY AT {MAX_EPOCHS_NR} EPOCHS: {accuracies_np.mean()} (Std. Dev.: {accuracies_np.std()}) over {len(accuracies_np)} runs"
)
print(
    f"AVERAGE ACCURACY AT HINTON POINT ({int(1425)} EPOCHS): {hp_acc_list.mean()} (Std. Dev.: {hp_acc_list.std()}) over {len(hp_acc_list)} runs"
)
print(" ")
print(
    f"CONVERGED AT {MAX_EPOCHS_NR} EPOCHS: {len(conv_epoch_np)} over {len(accuracies_np)} runs"
)
print(
    f"CONVERGED AT HINTON POINT ({int(1425)} EPOCHS): {(hp_acc_list == 1.0).sum()} over {len(accuracies_np)} runs"
)
print(" ")
if len(conv_epoch_np) > 0:
    print(
        f"AVERAGE EPOCHS UNTIL CONVERGENCE: {conv_epoch_np.mean()} (Std. Dev.: {conv_epoch_np.std()})"
    )
