#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, List
from copy import deepcopy
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
import torch as th
from torch import Tensor
from ebtorch.nn.utils import argser_f, emplace_kv
from src.util import no_op

# Run training until (hopeful) convergence; gather statistics
def train_diag_aio(
    model,
    dataloader: DataLoader,
    max_epochs_nr: int,
    loss,
    optimizer_fx,
    optimizer_dict: dict,
    device,
) -> Tuple[Tuple, Tuple[float, ...]]:

    optimizer_params: dict = emplace_kv(optimizer_dict, "params", model.parameters())
    optimizer = argser_f(optimizer_fx, optimizer_params)()

    losses: list = []
    accuracies: List[float] = []

    # Move model to device
    model = model.to(device)

    # Put model in training mode
    model.train()

    # Iterate over epochs
    _: int
    for _ in range(max_epochs_nr):

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


def train_one_run_diag(
    run_idx: int,
    model,
    init_fx,
    max_epochs_nr: int,
    loss_fx,
    optimizer_fx,
    optimizer_params: dict,
    device,
    dataloader: DataLoader,
) -> Tuple[float, ...]:
    _ = run_idx
    init_fx(model)
    _, to_be_ret = train_diag_aio(
        model,
        dataloader,
        max_epochs_nr,
        loss_fx,
        optimizer_fx,
        optimizer_params,
        device,
    )
    return to_be_ret


def train_parallel_runs_diag(
    runs_nr: int,
    processes_nr: int,
    model,
    init_fx,
    max_epochs_nr: int,
    loss_fx,
    optimizer_fx,
    optimizer_params: dict,
    device,
    dataloader: DataLoader,
) -> List[Tuple[float, ...]]:
    ret_list = Parallel(n_jobs=processes_nr)(
        delayed(train_one_run_diag)(
            run_idx,
            model,
            init_fx,
            max_epochs_nr,
            loss_fx,
            optimizer_fx,
            optimizer_params,
            device,
            dataloader,
        )
        for run_idx in range(runs_nr)
    )
    return ret_list


def train_and_gather_stats(
    runs_nr: int,
    processes_nr: int,
    model,
    init_fx,
    max_epochs_nr: int,
    loss_fx,
    optimizer_fx,
    optimizer_params: dict,
    device,
    dataloader: DataLoader,
    hintonpoint: int = 1425,
) -> None:

    converged_epochs_list: List[int] = []
    accuracies_list: List[float] = []
    hintonpoint_acc_list: List[float] = []

    traced_execution = train_parallel_runs_diag(
        runs_nr,
        processes_nr,
        model,
        init_fx,
        max_epochs_nr,
        loss_fx,
        optimizer_fx,
        optimizer_params,
        device,
        dataloader,
    )

    for a in traced_execution:
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
        f"AVERAGE ACCURACY AT {max_epochs_nr} EPOCHS: {accuracies_np.mean()} (Std. Dev.: {accuracies_np.std()}) over {len(accuracies_np)} runs"
    )
    print(
        f"AVERAGE ACCURACY AT HINTON POINT ({int(hintonpoint)} EPOCHS): {hp_acc_list.mean()} (Std. Dev.: {hp_acc_list.std()}) over {len(hp_acc_list)} runs"
    )
    print(" ")
    print(
        f"CONVERGED AT {max_epochs_nr} EPOCHS: {len(conv_epoch_np)} over {len(accuracies_np)} runs"
    )
    print(
        f"CONVERGED AT HINTON POINT ({int(hintonpoint)} EPOCHS): {(hp_acc_list == 1.0).sum()} over {len(accuracies_np)} runs"
    )
    print(" ")
    if len(conv_epoch_np) > 0:
        print(
            f"AVERAGE EPOCHS UNTIL CONVERGENCE: {conv_epoch_np.mean()} (Std. Dev.: {conv_epoch_np.std()})"
        )


if __name__ == "__main__":
    no_op()
