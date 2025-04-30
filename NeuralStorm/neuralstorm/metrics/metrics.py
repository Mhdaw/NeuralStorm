import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_standard_metrics(predictions, target, epsilon=1e-10):
    """Calculates standard time series metrics (MAE, MSE, RMSE, MAPE).

    Args:
        predictions: Model predictions (batch_size, horizon, num_quantiles) or (batch_size, horizon, 1, num_quantiles).
        target: Ground truth values (batch_size, horizon, 1) or (batch_size, horizon, 1, 1).
        epsilon: Small value to avoid division by zero in MAPE calculation.

    Returns:
        A dictionary containing MAE, MSE, RMSE, and MAPE.
    """
    # Remove dimensions of size 1 before calculating errors to prevent mismatches
    y_true = target.squeeze()
    y_pred_mean = predictions[:, :, 0]  # the mean is at index 0

    # Calculate errors
    abs_error = torch.abs(y_true - y_pred_mean)
    sq_error = abs_error ** 2

    # Calculate metrics
    mae = torch.mean(abs_error)
    mse = torch.mean(sq_error)
    rmse = torch.sqrt(mse)
    mape = torch.mean(torch.div(abs_error, torch.abs(y_true) + epsilon)) * 100

    return {'mae': mae.item(), 'mse': mse.item(), 'rmse': rmse.item(), 'mape': mape.item()}

def map_to_severity(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """
    Maps continuous values to discrete severity levels based on boundaries.

    Args:
        values (torch.Tensor): Tensor of continuous values (e.g., customers_out).
                               Shape: (batch_size, horizon)
        boundaries (torch.Tensor): 1D Tensor of upper boundaries for severity levels.
                                   Must be sorted. (e.g., [10, 100, 1000])
                                   Values < boundaries[0] -> level 0
                                   boundaries[0] <= values < boundaries[1] -> level 1
                                   ...
                                   values >= boundaries[-1] -> level len(boundaries)

    Returns:
        torch.Tensor: Tensor of integer severity levels. Shape: (batch_size, horizon)
    """

    boundaries = boundaries.to(values.device)
    severity_levels = torch.bucketize(values, boundaries)
    return severity_levels.long() # Ensure integer type

def calculate_severity_accuracy(predictions: torch.Tensor,
                                target: torch.Tensor,
                                boundaries: torch.Tensor,
                                median_idx: int) -> dict:
    """
    Calculates point-wise severity accuracy per horizon step.

    Args:
        predictions (torch.Tensor): Model output tensor including mean and quantiles.
                                    Shape: (batch_size, horizon, num_quantiles + 1)
        target (torch.Tensor): Ground truth target values.
                               Shape: (batch_size, horizon, 1)
        boundaries (torch.Tensor): 1D Tensor of severity boundaries.
        median_idx (int): Index of the median quantile in the prediction tensor (e.g., 5 if mean is index 0).

    Returns:
        dict: Contains 'overall_severity_accuracy' (scalar) and
              'severity_accuracy_per_step' (Tensor, shape: (horizon,)).
    """
    if predictions.shape[1] != target.shape[1]:
         raise ValueError(f"Horizon dims must match between predictions ({predictions.shape[1]}) and target ({target.shape[1]})")
    if median_idx >= predictions.shape[2]:
        raise ValueError(f"median_idx ({median_idx}) out of bounds for predictions last dim ({predictions.shape[2]})")

    # Extract median prediction and target
    y_pred_median = predictions[:, :, median_idx].detach() # Shape: (batch, horizon)
    y_true = target.squeeze(-1).detach().float()          # Shape: (batch, horizon)

    # Map to severity levels
    pred_severity = map_to_severity(y_pred_median, boundaries) # (batch, horizon)
    actual_severity = map_to_severity(y_true, boundaries)      # (batch, horizon)

    # Calculate correctness (element-wise comparison)
    correct = (pred_severity == actual_severity).float() # (batch, horizon)

    # Calculate overall accuracy (mean over batch and horizon)
    overall_accuracy = torch.mean(correct).item() * 100 # Percentage

    # Calculate accuracy per horizon step (mean over batch dimension)
    accuracy_per_step = torch.mean(correct, dim=0) * 100 # Percentage, Shape: (horizon,)

    return {
        'overall_severity_accuracy': overall_accuracy,
        'severity_accuracy_per_step': accuracy_per_step,
        #'pred_severity': pred_severity,
        #'actual_severity': actual_severity
    }


def plot_severity_accuracy(accuracy_per_step: torch.Tensor, save_path: str = None):
    """
    Plots the severity accuracy over the forecast horizon.

    Args:
        accuracy_per_step (torch.Tensor): Accuracy for each step. Shape: (horizon,)
        save_path (str, optional): Path to save the plot image. If None, shows plot.
    """
    horizon = len(accuracy_per_step)
    steps = np.arange(horizon)
    accuracy_values = accuracy_per_step.cpu().numpy() # Move to CPU and convert to NumPy

    plt.figure(figsize=(10, 5))
    plt.plot(steps, accuracy_values, marker='o', linestyle='-')
    plt.title('Severity Level Prediction Accuracy over Forecast Horizon')
    plt.xlabel('Forecast Horizon Step')
    plt.ylabel('Accuracy (%)')
    plt.xticks(steps[::max(1, horizon//10)]) # Adjust tick frequency for readability
    plt.ylim(0, 105) # between 0 and 100
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Severity accuracy plot saved to {save_path}")
        plt.close() 
    else:
        plt.show()