import os
import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from ..configs.multichannel_config import MultiChannelTimesFMConfig
from .timesfm import PatchedTimeSeriesDecoder

class MultiChannelTimesFM(nn.Module):
    """
    Combines multiple univariate TimesFM models for multivariate input,
    optionally incorporating temporal and event features.

    Processes each input time series channel with a separate TimesFM instance
    and combines their outputs along with processed auxiliary features(temporal, Storm events)
    using an MLP combiner.
    """
    def __init__(self, config: MultiChannelTimesFMConfig):
        """
        Initializes the MultiChannelTimesFM model.

        Args:
            config: A MultiChannelTimesFMConfig dataclass instance containing all necessary
                    parameters for the base TimesFM model, multi-channel setup,
                    feature processors, and combiner.
        """
        super().__init__()
        self.config = config
        self.model_cfg = config.model_config
        self.num_input_channels = config.num_input_channels

        if self.config.save_safetensors:
          from safetensors.torch import save_file
          self.save_safetensors = True
        else:
          self.save_safetensors = False
        # Initialize TimesFM Models per Channel
        self.timesfm_models = nn.ModuleList([
            PatchedTimeSeriesDecoder(self.model_cfg)
            for _ in range(self.num_input_channels)
        ])

        # Calculate expected output features (quantiles + mean)
        self.num_output_features = len(self.model_cfg.quantiles) + 1
        self.model_output_horizon = self.model_cfg.horizon_len

        # Initialize Feature Processors
        self.temporal_processor, self.actual_temporal_dim = self._setup_temporal_processor(config)
        self.event_processor, self.actual_event_dim = self._setup_event_processor(config)

        # Initialize Combiner MLP
        self.combiner_mlp = self._setup_combiner_mlp(config)

        self._print_initialization_summary()

    def _setup_temporal_processor(self, config: MultiChannelTimesFMConfig) -> tuple[Optional[nn.Module], int]:
        """Initializes the temporal feature processor if enabled and valid."""
        use_temporal = config.use_temporal_processor and config.num_temporal_features > 0
        if use_temporal:
            print("Temporal Processor: Enabled")
            processor = TemporalProcessor(
                config.num_temporal_features,
                config.temporal_processor_output_dim,
                config.context_window
            )
            return processor, config.temporal_processor_output_dim
        else:
            print(f"Temporal Processor: Disabled (use={config.use_temporal_processor}, num_features={config.num_temporal_features})")
            return None, 0

    def _setup_event_processor(self, config: MultiChannelTimesFMConfig) -> tuple[Optional[nn.Module], int]:
        """Initializes the event feature processor if enabled and valid."""
        use_event = config.use_event_processor and config.num_event_features > 0
        if use_event:
            print("Event Processor: Enabled")
            processor = EventProcessor(
                config.num_event_features,
                config.event_processor_output_dim
            )
            return processor, config.event_processor_output_dim
        else:
            print(f"Event Processor: Disabled (use={config.use_event_processor}, num_features={config.num_event_features})")
            return None, 0

    def _setup_combiner_mlp(self, config: MultiChannelTimesFMConfig) -> nn.Module:
        """Initializes the MLP that combines channel outputs and features."""
        combiner_input_dim = (self.num_input_channels * self.num_output_features +
                              self.actual_temporal_dim +
                              self.actual_event_dim)

        if combiner_input_dim <= 0:
            raise ValueError(f"Combiner input dimension must be positive, but got {combiner_input_dim}. "
                             "Check channel counts and processor settings.")

        print(f"Combiner MLP Input Dim: {combiner_input_dim}")
        return nn.Sequential(
            nn.Linear(combiner_input_dim, config.combiner_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.combiner_hidden_dim, self.num_output_features)
        )

    def _print_initialization_summary(self):
        """Prints a summary of the initialized model configuration."""
        print("\n--- MultiChannelTimesFM Initialization Summary ---")
        print(f"  Base Model: TimesFM (PatchedTimeSeriesDecoder)")
        print(f"  Input Channels: {self.num_input_channels}")
        print(f"  Output Features (per channel & final): {self.num_output_features}")
        print(f"  Model Output Horizon: {self.model_output_horizon}")
        print(f"  Temporal Processor Output Dim: {self.actual_temporal_dim}")
        print(f"  Event Processor Output Dim: {self.actual_event_dim}")
        print("--------------------------------------------------\n")


    def _process_channels(self,
                          input_ts_channels: List[torch.Tensor],
                          input_padding_channels: List[torch.Tensor],
                          frequency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Processes each input channel through its dedicated TimesFM model.

        Handles NaN checks, model inference, horizon alignment, and masking.

        Args:
            input_ts_channels: List of tensors [ (B, L) ] for each channel.
            input_padding_channels: List of tensors [ (B, L) ] for each channel's padding.
            frequency: Tensor (B,) indicating frequency for each batch item.

        Returns:
            A tuple containing:
            - stacked_outputs (torch.Tensor): Outputs stacked across channels (B, H, Q, C).
            - active_channel_mask (torch.Tensor): Mask indicating active channels (B, C).
            - nan_detected (bool): Flag indicating if NaNs were detected and handled.
        """
        channel_outputs = []
        active_channel_mask_list = []
        nan_detected_in_channels = False
        active_channel_threshold = 1e-6 # Threshold to consider a channel active, its numerical but the goal is to mask missing data which works

        for i in range(self.num_input_channels):
          abs_sum_real_data = torch.sum(torch.abs(input_ts_channels[i] * (1 - input_padding_channels[i])), dim=1)
          is_active_channel = (abs_sum_real_data > active_channel_threshold).float()
          active_channel_mask_list.append(is_active_channel)

        active_channel_mask = torch.cat(active_channel_mask_list, dim=1)
        broadcast_mask = active_channel_mask.unsqueeze(1).unsqueeze(1)

        for i in range(self.num_input_channels):
          input_ts_i = input_ts_channels[i].squeeze(-1)
          input_padding_i = input_padding_channels[i].squeeze(-1)

          # Pre-Model Check
          if torch.isnan(input_ts_i).any() or torch.isinf(input_ts_i).any():
            #print(f"!!! WARNING: NaN/Inf detected in input_ts_i for channel {i} BEFORE TimesFM !!!")
            # a NaN tensor of the expected shape
            dummy_output = torch.full((self.batch_size, self.config.num_layers, self.model_output_horizon, self.num_output_features),
                                      float('nan'), device=input_ts_i.device)
            n_patches = (input_ts_i.shape[1] + self.model_cfg.patch_len -1) // self.model_cfg.patch_len # Estimate number of patches
            h_patch = self.model_cfg.patch_len
            forecast_i = torch.full((self.batch_size, self.model_output_horizon, self.num_output_features), float('nan'), device=input_ts_i.device)
            nan_detected_in_any_channel = True
            #print(f"    --> Using dummy NaN forecast for channel {i}")
          else:
            model_output_i = self.timesfm_models[i](
                input_ts = input_ts_i,
                input_padding=input_padding_i,
                freq=frequency
            )
            if torch.isnan(model_output_i).any() or torch.isinf(model_output_i).any():
              #print(f"!!! WARNING: NaN/Inf DETECTED in raw output of TimesFM channel {i} !!!")
              model_output_i = torch.nan_to_num(model_output_i, nan=0.0, posinf=1e6, neginf=-1e6)
              nan_detected_in_any_channel = True

            forecast_i = model_output_i[:, -1, :, :]

            if torch.isnan(forecast_i).any() or torch.isinf(forecast_i).any():
              #print(f"!!! WARNING: NaN/Inf DETECTED in forecast_i (extracted) for channel {i} !!!")
              forecast_i = torch.nan_to_num(forecast_i, nan=0.0, posinf=1e6, neginf=-1e6)
              nan_detected_in_any_channel = True

            if forecast_i.shape[1] != self.model_output_horizon:
              #print(f"Warning: Channel {i} forecast horizon ({forecast_i.shape[1]}) doesn't match expected ({self.model_output_horizon}). Check TimesFM config/output.")
              if forecast_i.shape[1] > self.model_output_horizon:
                forecast_i = forecast_i[:, :self.model_output_horizon, :]
              else:
                pad_size = self.model_output_horizon - forecast_i.shape[1]
                forecast_i = F.pad(forecast_i, (0, 0, 0, pad_size))
          channel_outputs.append(forecast_i)

        stacked_outputs = torch.stack(channel_outputs, dim=-1)
        masked_outputs = stacked_outputs * broadcast_mask
        if torch.isnan(masked_outputs).any():
          masked_outputs = torch.nan_to_num(masked_outputs, nan=0.0, posinf=1e6, neginf=-1e6)

        return stacked_outputs, masked_outputs, active_channel_mask, nan_detected_in_channels

    def _process_aux_features(self, batch: Dict[str, torch.Tensor]) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Processes temporal and event features if enabled."""
        processed_temporal, processed_events = None, None
        nan_detected = False

        # Temporal Features
        if self.temporal_processor is not None:
            temporal_features = batch.get('temporal_features')
            if temporal_features is None:
                 #print("!!! ERROR: Temporal processor enabled, but 'temporal_features' not found in batch !!!")
                 return torch.full((self.batch_size, self.model_output_horizon, self.num_output_features), float('nan'), device=temporal_features.device)
            if not torch.all(torch.isfinite(temporal_features)):
                 #print(f"!!! WARNING: NaN/Inf detected in temporal_features BEFORE processor !!!")
                 temporal_features = torch.nan_to_num(temporal_features, nan=0.0, posinf=1e6, neginf=-1e6)
                 nan_detected = True

            processed_temporal = self.temporal_processor(temporal_features) # Should output (B, D_temporal)

            if not torch.all(torch.isfinite(processed_temporal)):
                 #print(f"!!! WARNING: NaN/Inf detected in processed_temporal AFTER processor !!!")
                 processed_temporal = torch.nan_to_num(processed_temporal, nan=0.0, posinf=1e6, neginf=-1e6)
                 nan_detected = True

        # Event Features
        if self.event_processor is not None:
            event_features = batch.get('event_features')
            if event_features is None:
                 #print("Event processor enabled, but 'event_features' not found in batch!")
                 return torch.full((self.batch_size, self.model_output_horizon, self.num_output_features), float('nan'), device=event_features.device)
            if not torch.all(torch.isfinite(event_features)):
                 #print(f"!!! WARNING: NaN/Inf detected in event_features BEFORE processor !!!")
                 event_features = torch.nan_to_num(event_features, nan=0.0, posinf=1e6, neginf=-1e6)
                 nan_detected = True

            processed_events = self.event_processor(event_features) # Should output (B, D_event)

            if not torch.all(torch.isfinite(processed_events)):
                 #print(f"!!! WARNING: NaN/Inf detected in processed_events AFTER processor !!!")
                 processed_events = torch.nan_to_num(processed_events, nan=0.0, posinf=1e6, neginf=-1e6)
                 nan_detected = True

        if nan_detected:
            #print("    --> NaNs/Infs were handled in auxiliary feature processing.")

        return processed_temporal, processed_events

    def _multi_variate_forecast(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        input_ts = batch['input_ts']
        input_padding = batch['input_padding']
        frequency = batch['frequency']

        batch_size = input_ts.shape[0]
        self.batch_size = batch_size # easy fix
        if torch.isnan(input_ts).any() or torch.isinf(input_ts).any():
          #print("!!! FATAL: NaN/Inf found in input_ts BATCH INPUT !!!")
          return torch.full((batch_size, self.model_output_horizon, self.num_output_features), float('nan'), device=input_ts.device) # Return NaNs immediately

        input_ts_channels = input_ts.split(1, dim=-1)
        input_padding_channels = input_padding.split(1, dim=-1)

        stacked_outputs, _, _, _ = self._process_channels(input_ts_channels,
                                                          input_padding_channels,
                                                          frequency)
        unstacked_outputs = torch.unbind(stacked_outputs, dim=-1)
        return list(unstacked_outputs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs the forward pass of the MultiChannelTimesFM model.

        Args:
            batch: A dictionary containing the input tensors:
                - 'input_ts': The multivariate time series (B, L, C).
                - 'input_padding': Padding mask for the input time series (B, L, C).
                - 'frequency': Frequency indicator for each batch item (B,).
                - 'temporal_features': Optional temporal context features (B, L_ctx, F_t).
                - 'event_features': Optional aggregated event features (B, F_e).

        Returns:
            torch.Tensor: The final combined forecast (B, H, Q).
        """
        input_ts = batch['input_ts']
        input_padding = batch['input_padding']
        frequency = batch['frequency']

        batch_size = input_ts.shape[0]
        if torch.isnan(input_ts).any() or torch.isinf(input_ts).any():
             #print("!!! FATAL: NaN/Inf found in input_ts BATCH INPUT !!!")
             return torch.full((batch_size, self.model_output_horizon, self.num_output_features), float('nan'), device=input_ts.device) # Return NaNs immediately

        input_ts_channels = input_ts.split(1, dim=-1)
        input_padding_channels = input_padding.split(1, dim=-1)

        self.batch_size = batch_size

        stacked_outputs, masked_outputs, active_channel_mask, nan_detected_in_channels = self._process_channels(input_ts_channels,
                                                                                                                input_padding_channels,
                                                                                                                frequency)
        reshaped_for_mlp = masked_outputs.permute(0, 1, 3, 2)
        mlp_input_base = reshaped_for_mlp.reshape(batch_size, self.model_output_horizon, -1)
        features_to_concat = [mlp_input_base]

        if self.config.use_temporal_processor and self.temporal_processor is not None:
          processed_temporal, processed_events = self._process_aux_features(batch)

          temporal_repeated = processed_temporal.unsqueeze(1).expand(-1, self.model_output_horizon, -1)
          features_to_concat.append(temporal_repeated)

          events_repeated = processed_events.unsqueeze(1).expand(-1, self.model_output_horizon, -1)
          features_to_concat.append(events_repeated)

        try:
            mlp_input = torch.cat(features_to_concat, dim=-1)
        except Exception as e:
            #print(f"!!! ERROR during torch.cat for MLP input: {e} !!!")
            #print("Shapes of features being concatenated:")
            #for i, feat in enumerate(features_to_concat):
                #print(f"  Feature {i}: {feat.shape}")
            return torch.full((batch_size, self.model_output_horizon, self.num_output_features), float('nan'), device=input_ts.device)
        if torch.isnan(mlp_input).any() or torch.isinf(mlp_input).any():
             #print(f"!!! WARNING: NaN/Inf DETECTED in mlp_input BEFORE combiner MLP !!!")
             mlp_input = torch.nan_to_num(mlp_input, nan=0.0, posinf=1e6, neginf=-1e6)

        final_forecast = self.combiner_mlp(mlp_input)
        if torch.isnan(final_forecast).any() or torch.isinf(final_forecast).any():
          final_forecast = torch.nan_to_num(final_forecast, nan=0.0, posinf=1e6, neginf=-1e6)
        return final_forecast

    def _save_checkpoint_safetensors(self, path: str):
        """Saves the model's weights in .safetensors ."""
        if self.save_safetensors:
          state_dict = self.state_dict()
          try:
            save_file(state_dict, path)
            print(f"saved model weights to: {path}")
          except Exception as e:
            print(f"Error saving model weights to {path}: {e}")
    def _save_checkpoint_torch(self, path: str):
        """Saves the model's weights in .pt ."""
        try:
          torch.save(self.state_dict(), path)
          print(f"model saved to: {path}")
        except Exception as e:
              print(f"Error saving model weights to {path}: {e}")

    def save_checkpoint(self):
        """Saves checkpoint"""
        folder_name = "ModelWeights"
        model_name = "MultiVariateTimesFM"

        try:
          os.makedirs(folder_name, exist_ok=True)
          if self.save_safetensors:
            file_extension = ".safetensors"
            path = os.path.join(folder_name, f"{model_name}{file_extension}")
            self._save_checkpoint_safetensors(path)
          else:
            file_extension = ".pt" # .pth
            path = os.path.join(folder_name, f"{model_name}{file_extension}")
            self._save_checkpoint_torch(path)
        except Exception as e:
          print(f"An error occurred during checkpoint saving setup: {e}")


class TemporalProcessor(nn.Module):
    """Processes context temporal features for the combiner."""
    def __init__(self, num_temporal_features: int, output_dim: int, context_window: int):
        super().__init__()
        self.num_temporal_features = num_temporal_features
        self.output_dim = output_dim
        self.context_window = context_window

        self.processor = nn.Sequential(
            nn.Linear(num_temporal_features, output_dim),
            nn.ReLU()
        )
        print(f"TemporalProcessor initialized: Input Features={num_temporal_features}, Output Dim={output_dim}, Pooling=Mean")

    def forward(self, temporal_features_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features_context (torch.Tensor): Shape (B, Context, N_t)
        Returns:
            torch.Tensor: Processed features, Shape (B, output_dim)
        """
        # Mean pooling over the context window dimension
        # Ensure float for mean calculation if input is int
        pooled_features = temporal_features_context.float().mean(dim=1) # Shape (B, N_t)
        processed = self.processor(pooled_features) # Shape (B, output_dim)
        return processed

class EventProcessor(nn.Module):
    """Processes aggregated event features for the combiner."""
    def __init__(self, num_event_features: int, output_dim: int):
        super().__init__()
        self.num_event_features = num_event_features
        self.output_dim = output_dim

        # Example: Simple Linear layer
        self.processor = nn.Sequential(
            nn.Linear(num_event_features, output_dim),
            nn.ReLU()
        )
        print(f"EventProcessor initialized: Input Features={num_event_features}, Output Dim={output_dim}")

    def forward(self, event_features_aggregated: torch.Tensor) -> torch.Tensor:
        """
        Args:
            event_features_aggregated (torch.Tensor): Shape (B, N_e)
        Returns:
            torch.Tensor: Processed features, Shape (B, output_dim)
        """
        processed = self.processor(event_features_aggregated) # Shape (B, output_dim)
        return processed
