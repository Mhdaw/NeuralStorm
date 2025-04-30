"""NeuralStorm - Predicting power outages from extreme weather events."""

from .timesfm import (
    TimesFMConfig,
    MultiChannelTimesFMConfig,
    PatchedTimeSeriesDecoder,
    MultiChannelTimesFM
)

from .dataset import StormTimeSeriesDataset
from .loss import QuantileLoss
from .metrics import (
    calculate_standard_metrics,
    map_to_severity,
    calculate_severity_accuracy,
    plot_severity_accuracy
)

__version__ = "0.1.0"

__all__ = [
    'TimesFMConfig',
    'MultiChannelTimesFMConfig',
    'PatchedTimeSeriesDecoder',
    'MultiChannelTimesFM',
    'StormTimeSeriesDataset',
    'QuantileLoss',
    'calculate_standard_metrics',
    'map_to_severity',
    'calculate_severity_accuracy',
    'plot_severity_accuracy'
]