import torch
from torch import nn
import torch.nn.functional as F

class QuantileLoss(nn.Module):
    def __init__(self, quantiles, device=None):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles))

    def forward(self, predictions, target):
        # predictions shape: [batch_size, horizon_length, quantiles]
        target = target.unsqueeze(-1)
        predictions_quantiles = predictions[:, :, 1:]
        errors = target - predictions_quantiles

        losses = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors
        )
        return losses.mean()