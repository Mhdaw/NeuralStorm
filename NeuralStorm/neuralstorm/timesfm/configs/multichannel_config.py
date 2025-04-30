import dataclasses
from typing import List, Tuple
from .timesfm_config import TimesFMConfig, create_quantiles

@dataclasses.dataclass
class MultiChannelTimesFMConfig:
  """Overall configuration for the model and training setup."""

  # Nested Model Config
  model_config: TimesFMConfig = dataclasses.field(default_factory=TimesFMConfig)

  num_input_channels: int =6
  num_temporal_features: int = 4
  num_event_features: int = 55
  context_window: int = 64
  combiner_hidden_dim: int = 128
  temporal_processor_output_dim: int = 32
  event_processor_output_dim: int = 32
  use_temporal_processor: bool = True
  use_event_processor: bool = True

  save_safetensors: bool = True