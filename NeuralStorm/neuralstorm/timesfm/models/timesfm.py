import math
import torch
from torch import nn
import torch.nn.functional as F
from ..configs.timesfm_config import TimesFMConfig
from .components import (
    ResidualBlock,
    StackedDecoder,
    PositionalEmbedding,
    _masked_mean_std,
    _shift_padded_seq
)

class PatchedTimeSeriesDecoder(nn.Module):
  """Patched time-series decoder."""

  def __init__(self, config: TimesFMConfig):
    super().__init__()
    self.config = config
    self.input_ff_layer = ResidualBlock(
        input_dims=2 * config.patch_len,
        output_dims=config.hidden_size,
        hidden_dims=config.intermediate_size,
    )
    self.freq_emb = nn.Embedding(num_embeddings=3,
                                 embedding_dim=config.hidden_size)
    self.horizon_ff_layer = ResidualBlock(
        input_dims=config.hidden_size,
        output_dims=config.horizon_len * (1 + len(config.quantiles)),
        hidden_dims=config.intermediate_size,
    )
    self.stacked_transformer = StackedDecoder(
        hidden_size=self.config.hidden_size,
        intermediate_size=self.config.intermediate_size,
        num_heads=self.config.num_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        num_layers=self.config.num_layers,
        rms_norm_eps=self.config.rms_norm_eps,
    )
    if self.config.use_positional_embedding:
      self.position_emb = PositionalEmbedding(self.config.hidden_size)

  def _forward_transform(
      self, inputs: torch.Tensor, patched_pads: torch.Tensor
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Input is of shape [B, N, P]."""
    mu, sigma = _masked_mean_std(inputs, patched_pads)
    sigma = torch.where(
        sigma < self.config.tolerance,
        torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
        sigma,
    )
    sigma_safe = torch.clamp(sigma, min=1e-6)

    # Normalize each patch
    outputs = (inputs - mu[:, None, None]) / sigma_safe[:, None, None]
    outputs = torch.where(
        torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
        torch.tensor(self.config.pad_val,
                     dtype=outputs.dtype,
                     device=outputs.device),
        outputs,
    )
    return outputs, (mu, sigma_safe)

  def _reverse_transform(
      self, outputs: torch.Tensor, stats: tuple[torch.Tensor,
                                                torch.Tensor]) -> torch.Tensor:
    """Output is of shape [B, N, P, Q]."""
    mu, sigma = stats
    return outputs * sigma[:, None, None, None] + mu[:, None, None, None]

  def _preprocess_input(
      self,
      input_ts: torch.Tensor,
      input_padding: torch.Tensor,
  ) -> tuple[
      torch.Tensor,
      torch.Tensor,
      tuple[torch.Tensor, torch.Tensor] | None,
      torch.Tensor,
  ]:
    """Preprocess input for stacked transformer."""

    # Reshape into patches (using view for efficiency)
    bsize = input_ts.shape[0]
    patched_inputs = input_ts.view(bsize, -1, self.config.patch_len)
    patched_pads = input_padding.view(bsize, -1, self.config.patch_len)

    patched_inputs = torch.where(
        torch.abs(patched_pads - 1.0) < self.config.tolerance,
        torch.tensor(0.0,
                     dtype=patched_inputs.dtype,
                     device=patched_inputs.device),
        patched_inputs,
    )
    patched_pads = torch.where(
        torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
        torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
        patched_pads,
    )
    patched_inputs, stats = self._forward_transform(patched_inputs,
                                                    patched_pads)

    # B x N x D
    patched_inputs = patched_inputs * (1.0 - patched_pads)
    concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
    model_input = self.input_ff_layer(concat_inputs)

    # A patch should not be padded even if there is at least one zero.
    patched_padding = torch.min(patched_pads,
                                dim=-1)[0]  # Get the values from the min result
    if self.config.use_positional_embedding:
      pos_emb = self.position_emb(model_input.shape[1]).to(model_input.device)
      pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
      pos_emb = _shift_padded_seq(patched_padding, pos_emb)
      model_input += pos_emb

    return model_input, patched_padding, stats, patched_inputs

  def _postprocess_output(
      self,
      model_output: torch.Tensor,
      num_outputs: int,
      stats: tuple[torch.Tensor, torch.Tensor],
  ) -> torch.Tensor:
    """Postprocess output of stacked transformer."""

    # B x N x (H.Q)
    output_ts = self.horizon_ff_layer(model_output)

    # Reshape using view
    b, n, _ = output_ts.shape
    output_ts = output_ts.view(b, n, self.config.horizon_len, num_outputs)

    return self._reverse_transform(output_ts, stats)

  def forward(
      self,
      input_ts: torch.Tensor,
      input_padding: torch.LongTensor,
      freq: torch.Tensor,
  ) -> torch.Tensor:
    num_outputs = len(self.config.quantiles) + 1
    model_input, patched_padding, stats, _ = self._preprocess_input(
        input_ts=input_ts,
        input_padding=input_padding,
    )
    f_emb = self.freq_emb(freq)  # B x 1 x D
    model_input += f_emb.unsqueeze(1)
    model_output = self.stacked_transformer(model_input, patched_padding)

    output_ts = self._postprocess_output(model_output, num_outputs, stats)
    return output_ts

  def decode(
      self,
      input_ts: torch.Tensor,
      paddings: torch.Tensor,
      freq: torch.LongTensor,
      horizon_len: int,
      output_patch_len: int | None = None,
      max_len: int | None = None,
      return_forecast_on_context: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Auto-regressive decoding without caching.

    Args:
      input_ts: input time-series and paddings. Time-series shape B x C.
      paddings: padding shape B x (C + H) where H is the prediction length.
      freq: frequency shape B x 1
      horizon_len: prediction length.
      output_patch_len: output length to be fetched from one step of
        auto-regressive decoding.
      max_len: maximum training context length.
      return_forecast_on_context: whether to return the model forecast on the
        context except the first input patch.

    Returns:
      Tuple of two forecasting results:
      - Point (mean) output predictions as a tensor with shape B x H'.
      - Full predictions (mean and quantiles) as a tensor with shape
        B x H' x (1 + # quantiles).
      In particular, if return_forecast_on_context is True, H' is H plus
      the forecastable context length, i.e. context_len - (first) patch_len.
    """
    final_out = input_ts
    context_len = final_out.shape[1]
    full_outputs = []
    if max_len is None:
      max_len = context_len
    if paddings.shape[1] != final_out.shape[1] + horizon_len:
      raise ValueError(
          "Length of paddings must match length of input + horizon_len:"
          f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}")
    if output_patch_len is None:
      output_patch_len = self.config.horizon_len
    num_decode_patches = (horizon_len + output_patch_len -
                          1) // output_patch_len
    for step_index in range(num_decode_patches):
      current_padding = paddings[:, 0:final_out.shape[1]]
      input_ts = final_out[:, -max_len:]
      input_padding = current_padding[:, -max_len:]
      fprop_outputs = self(input_ts, input_padding, freq)
      if return_forecast_on_context and step_index == 0:
        # For the first decodings step, collect the model forecast on the
        # context except the unavailable first input batch forecast.
        new_full_ts = fprop_outputs[:, 0:-1, 0:self.config.patch_len, :]
        new_full_ts = new_full_ts.reshape(new_full_ts.size(0), -1,
                                          new_full_ts.size(3))

        full_outputs.append(new_full_ts)

      # (full batch, last patch, output_patch_len, index of mean forecast = 0)
      new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
      new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
      # (full batch, last patch, output_patch_len, all output indices)
      full_outputs.append(new_full_ts)
      final_out = torch.concatenate([final_out, new_ts], axis=-1)

    if return_forecast_on_context:
      # `full_outputs` indexing starts at after the first input patch.
      full_outputs = torch.concatenate(
          full_outputs,
          axis=1)[:, :(context_len - self.config.patch_len + horizon_len), :]
    else:
      # `full_outputs` indexing starts at the forecast horizon.
      full_outputs = torch.concatenate(full_outputs, axis=1)[:,
                                                             0:horizon_len, :]

    return (full_outputs[:, :, 0], full_outputs)