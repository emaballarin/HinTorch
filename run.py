#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Global imports
import numpy  # Load MKL!
from torch import optim
import torch.nn.functional as F

# %% Local imports
from src.data import train_unbalanced_dl, train_balanced_dl
from src.architectures import model_original, model_improved, model_relu, model_mish
from src.init import modern_init_, original_init_
from src.procedures import train_and_gather_stats

# %% Hyperparameters
MAX_EPOCHS_NR: int = int(5e4)
LOSS = F.mse_loss
OPTIMIZER_FX = optim.SGD
OPTIMIZER_PARAMS: dict = {"params": None, "lr": 0.1, "momentum": 0.9}
DEVICE = "cpu"

SAMPLE_SIZE: int = 500

MAX_EPOCHS_NR: int = max(MAX_EPOCHS_NR, 1426)

PROCESSES: int = 24

if DEVICE != "cpu":
    PROCESSES = 1

# %% RUNS
if __name__ == "__main__":

    # %% RUN 1:
    print(" ")

    print("~~~ MODEL: Original | INIT: Original | DATA: Imbalanced ~~~")
    train_and_gather_stats(
        SAMPLE_SIZE,
        PROCESSES,
        model_original,
        original_init_,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
        train_unbalanced_dl,
    )

    print(" ")
    print(" ")

    # %% RUN 2:
    print(" ")

    print("~~~ MODEL: Original | INIT: Original | DATA: Balanced ~~~")
    train_and_gather_stats(
        SAMPLE_SIZE,
        PROCESSES,
        model_original,
        original_init_,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
        train_balanced_dl,
    )

    print(" ")
    print(" ")

    # %% RUN 3:
    print(" ")

    print("~~~ MODEL: Improved (Tnah/Sigmoid) | INIT: Modern | DATA: Balanced ~~~")
    train_and_gather_stats(
        SAMPLE_SIZE,
        PROCESSES,
        model_improved,
        modern_init_,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
        train_balanced_dl,
    )

    print(" ")
    print(" ")

    # %% RUN 4:
    print(" ")

    print("~~~ MODEL: ReLU/Sigmoid | INIT: Modern | DATA: Balanced ~~~")
    train_and_gather_stats(
        SAMPLE_SIZE,
        PROCESSES,
        model_relu,
        modern_init_,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
        train_balanced_dl,
    )

    print(" ")
    print(" ")

    # %% RUN 5:
    print(" ")

    print("~~~ MODEL: Mish/Sigmoid | INIT: Modern | DATA: Balanced ~~~")
    train_and_gather_stats(
        SAMPLE_SIZE,
        PROCESSES,
        model_mish,
        modern_init_,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
        train_balanced_dl,
    )

    print(" ")
    print(" ")

    # %% RUN 6:
    print(" ")

    print("~~~ MODEL: Mish/Sigmoid | INIT: Modern | DATA: Imbalanced ~~~")
    train_and_gather_stats(
        SAMPLE_SIZE,
        PROCESSES,
        model_mish,
        modern_init_,
        MAX_EPOCHS_NR,
        LOSS,
        OPTIMIZER_FX,
        OPTIMIZER_PARAMS,
        DEVICE,
        train_unbalanced_dl,
    )

    print(" ")
    print(" ")
