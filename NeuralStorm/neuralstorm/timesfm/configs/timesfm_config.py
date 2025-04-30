import dataclasses
from typing import List, Tuple

def create_quantiles() -> list[float]:
    """Create default quantile values."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

@dataclasses.dataclass
class TimesFMConfig:
    """Config for initializing timesfm patched_decoder class."""

    # The number of blocks in the model
    num_layers: int = 20
    # The number of attention heads used in the attention layers
    num_heads: int = 16
    # The number of key-value heads for implementing attention
    num_kv_heads: int = 16
    # The hidden size of the model
    hidden_size: int = 1280
    # The dimension of the MLP representations
    intermediate_size: int = 1280
    # The number of head dimensions
    head_dim: int = 80
    # The epsilon used by the rms normalization layers
    rms_norm_eps: float = 1e-6
    # Patch length
    patch_len: int = 32
    # Horizon length
    horizon_len: int = 128
    # Quantiles
    quantiles: List[float] = dataclasses.field(default_factory=create_quantiles)
    # Padding value
    pad_val: float = 1123581321.0
    # Tolerance
    tolerance: float = 1e-6
    # The dtype of the weights
    dtype: str = "bfloat32"
    # Whether to use positional embedding
    use_positional_embedding: bool = True