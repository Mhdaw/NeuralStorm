import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_standard_metrics(predictions, target, epsilon=1e-10):
    """Calculates standard time series metrics, returns tensors."""
    if predictions.shape[-1] == 1:
        y_pred_point = predictions.squeeze(-1)
    elif predictions.shape[-1] > 1:
        y_pred_point = predictions[..., 0]
    else:
         raise ValueError("Prediction tensor has unexpected shape.")
    y_true = target.squeeze(-1)
    if y_pred_point.shape != y_true.shape:
         raise ValueError(f"Shape mismatch: Pred {y_pred_point.shape}, Target {y_true.shape}")

    abs_error = torch.abs(y_true - y_pred_point)
    sq_error = abs_error ** 2

    # Calculate metrics as tensors (NO .item())
    mae = torch.mean(abs_error).detach()
    mse = torch.mean(sq_error).detach()
    rmse = torch.sqrt(mse).detach()
    mape = (torch.mean(torch.div(abs_error, torch.abs(y_true) + epsilon)) * 100).detach()

    # Return dictionary of tensors
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}


def map_to_severity(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    boundaries = boundaries.to(values.device)
    severity_levels = torch.bucketize(values, boundaries, right=False)
    return severity_levels.long()

def calculate_severity_accuracy(predictions: torch.Tensor,
                                target: torch.Tensor,
                                boundaries: torch.Tensor,
                                median_idx: int) -> dict:
    """Calculates severity accuracy, returns tensors."""
    if predictions.shape[1] != target.shape[1]:
        raise ValueError(f"Horizon dims mismatch: Pred {predictions.shape[1]}, Target {target.shape[1]}")
    if median_idx >= predictions.shape[2]:
        raise ValueError(f"median_idx ({median_idx}) out of bounds: {predictions.shape[2]}")

    y_pred_median = predictions[:, :, median_idx].detach()
    y_true = target.squeeze(-1).detach().float()

    pred_severity = map_to_severity(y_pred_median, boundaries)
    actual_severity = map_to_severity(y_true, boundaries)

    correct = (pred_severity == actual_severity).float()

    # Return overall accuracy as a tensor (NO .item())
    overall_accuracy = (torch.mean(correct) * 100).detach()
    # Keep accuracy_per_step as a tensor (already was)
    accuracy_per_step = (torch.mean(correct, dim=0) * 100).detach()

    return {
        'overall_severity_accuracy': overall_accuracy, # tensor (scalar)
        'severity_accuracy_per_step': accuracy_per_step # tensor (horizon,)
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
