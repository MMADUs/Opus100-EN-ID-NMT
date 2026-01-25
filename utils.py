# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from typing import Dict, Optional


def is_improvement(value: float, best: float, mode: str, epsilon: float) -> bool:
    """
    Check if a metric value represents an improvement over the best value.
    """
    if mode == "min":
        return (best - value) > epsilon
    else:
        return (value - best) > epsilon


class TrainCheckpoint:
    """
    Tracks the best metric value seen during training and automatically
    saves the model state whenever the metric improves beyond the epsilon threshold.

    Args:
        filepath (str): Path where checkpoint files will be saved
        mode (str): Optimization mode, either "min" or "max"
        epsilon (float): Minimum improvement threshold

    Raises:
        AssertionError: If mode is not "min" or "max"
    """

    def __init__(self, filepath: str, mode: str = "min", epsilon: float = 0.0) -> None:
        assert mode in {"min", "max"}

        self.filepath = filepath
        self.mode = mode
        self.epsilon = epsilon
        self.reinit()

    def reinit(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else -float("inf")

    def step(self, value: float, checkpoint_dict: Dict) -> None:
        """
        Args:
            value (float): Current metric value to evaluate
            checkpoint_dict (Dict): Dictionary containing model state, optimizer state, etc. to save
        """
        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            self.best_value = value
            torch.save(checkpoint_dict, self.filepath)


class EarlyStopping:
    """
    Monitors a metric value and triggers early stopping when the metric
    fails to improve for a specified number of epochs (patience).

    Args:
        patience (int): Number of epochs with no improvement before stopping
        epsilon (float): Minimum improvement threshold
        mode (str): Optimization mode, either "min" or "max"

    Raises:
        AssertionError: If mode is not "min" or "max"
    """

    def __init__(self, patience: int, epsilon: float = 1e-4, mode: str = "min") -> None:
        assert mode in {"min", "max"}

        self.patience = patience
        self.epsilon = epsilon
        self.mode = mode
        self.reinit()

    def reinit(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else -float("inf")
        self.wait = 0

    def step(self, value: float) -> bool:
        """
        Args:
            value (float): Current metric value to evaluate

        Returns:
            bool: True if training should stop, False otherwise
        """
        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            print(f"* metrics improved from {self.best_value:.6f} to {value:.6f}")
            self.best_value = value
            self.wait = 0
            return False
        else:
            print(f"metrics did not improve from {self.best_value:.6f}")

        self.wait += 1
        early_stop = self.wait >= self.patience
        if early_stop:
            print(f"Early stopping, no improvement in the last {self.patience} epochs")

        return early_stop


class ReduceLROnPlateau:
    """
    Learning rate scheduling that reduces the optimizer's learning rate
    by a factor when a metric fails to improve for a specified number of epochs.
    Includes cooldown period to avoid thrashing and minimum learning rate boundary.

    Args:
        factor (float): Multiplicative factor for learning rate reduction (e.g., 0.1)
        patience (int): Number of epochs to wait before reducing LR
        cooldown (int): Number of epochs to wait after reduction before next reduction
        mode (str): "min" for minimization or "max" for maximization, default "min"
        epsilon (float): Minimum improvement threshold, default 1e-4
        min_lr (float): Minimum learning rate boundary, default 1e-6

    Raises:
        AssertionError: If mode is not "min" or "max"
    """

    def __init__(
        self,
        factor: float,
        patience: int,
        cooldown: int,
        mode: str = "min",
        epsilon: float = 1e-4,
        min_lr: float = 1e-6,
    ) -> None:
        assert mode in {"min", "max"}

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.epsilon = epsilon
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.reinit()

    def reinit(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else -float("inf")
        self.wait = 0
        self.cooldown_counter = 0

    def in_cooldown(self) -> bool:
        return self.cooldown_counter > 0

    def step(self, value: float, optimizer_dict: Dict) -> None:
        """
        Args:
            value (float): Current metric value to evaluate
            optimizer_dict (Dict): Dictionary containing optimizer(s) to update learning rates
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        if is_improvement(value, self.best_value, self.mode, self.epsilon):
            self.best_value = value
            self.wait = 0
            return

        self.wait += 1

        if self.wait >= self.patience:
            for optimizer in optimizer_dict.values():
                for group in optimizer.param_groups:
                    old_lr = group["lr"]
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    group["lr"] = new_lr

            self.wait = 0
            self.cooldown_counter = self.cooldown
            print(f"Reducing LR in the next epoch from {old_lr:.6f} to {new_lr:.6f}")


class TrainingCallback:
    """
    Unified callback handler that orchestrates callback execution
    during training and manages their interdependencies.

    Args:
        checkpoint (TrainCheckpoint, optional): Checkpoint handler, default None
        early_stop (EarlyStopping, optional): Early stopping handler, default None
        reduce_lr (ReduceLROnPlateau, optional): Learning rate scheduler, default None
    """

    def __init__(
        self,
        checkpoint: Optional[TrainCheckpoint] = None,
        early_stop: Optional[EarlyStopping] = None,
        reduce_lr: Optional[ReduceLROnPlateau] = None,
    ) -> None:
        self.cp = checkpoint
        self.es = early_stop
        self.rlr = reduce_lr

    def init(self) -> None:
        if self.cp:
            self.cp.reinit()
        if self.es:
            self.es.reinit()
        if self.rlr:
            self.rlr.reinit()

    def step(
        self,
        monitor_value: float,
        epoch: int,
        model_dict: Optional[Dict] = None,
        optimizer_dict: Optional[Dict] = None,
    ) -> bool:
        """
        Execute callbacks for the current training step.

        Args:
            monitor_value (float): The metric value to monitor
            epoch (int): Current epoch number
            model_dict (Dict, optional): Model state dict for checkpointing, default None
            optimizer_dict (Dict, optional): Optimizer state dict for checkpointing, default None

        Returns:
            bool: True if training should stop (early stopping triggered), False otherwise
        """
        # checkpoint
        if self.cp and model_dict:
            checkpoint_dict = {"model": model_dict, "epoch": epoch}

            if optimizer_dict:
                checkpoint_dict["optimizer"] = {
                    k: v.state_dict() if hasattr(v, "state_dict") else v
                    for k, v in optimizer_dict.items()
                }

            self.cp.step(monitor_value, checkpoint_dict)

        # lr scheduler
        if self.rlr and optimizer_dict:
            self.rlr.step(monitor_value, optimizer_dict)

        # early stop
        if self.es:
            early_stop = self.es.step(monitor_value)

            if self.rlr:
                if self.rlr.in_cooldown():
                    return False

            return early_stop

        return False


def time_formatter(sec_elapsed: float) -> str:
    """
    Format elapsed time in seconds to human-readable HH:MM:SS format.

    Args:
        sec_elapsed (float): Elapsed time in seconds

    Returns:
        str: Formatted time string in the format "H:M:S"
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"
