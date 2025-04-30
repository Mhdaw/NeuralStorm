import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Optional

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os


class StormTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for multivariate time series forecasting across multiple counties.

    Generates sequences of (input, target) pairs along with processed
    temporal and event features for each county independently.
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 context_window: int,
                 horizon: int,
                 input_cols: List[str],
                 target_col: str,
                 temporal_cols: List[str],
                 event_cols: List[str],
                 fips_col: str = 'fips_code',
                 time_col: str = 'time',
                 frequency: Optional[int] = 0
                ):
        """
        Args:
            dataframe (pd.DataFrame): Sorted DataFrame with time series data for all counties.
                                      MUST be sorted by fips_col and then time_col.
            context_window (int): Length of the input sequence (history).
            horizon (int): Length of the output sequence to predict (future).
            input_cols (List[str]): List of column names to use as input time series features.
            target_col (str): Name of the target column.
            temporal_cols (List[str]): List of column names for temporal features.
            event_cols (List[str]): List of column names for event features.
            fips_col (str): Column name identifying the county/group.
            time_col (str): Column name for the timestamp.
            frequency (Optional[int]): Frequency type (0, 1, or 2).

        """
        super().__init__()
        if len(dataframe) == 0:
            raise ValueError("Input DataFrame is empty.")
        if frequency not in [0, 1, 2]:
            raise ValueError("Invalid frequency value. Must be 0, 1, or 2.")

        self.df = dataframe
        self.context_window = context_window
        self.horizon = horizon
        self.input_cols = input_cols
        self.target_col = target_col
        self.temporal_cols = temporal_cols
        self.event_cols = event_cols
        self.fips_col = fips_col
        self.time_col = time_col
        self.frequency = frequency

        # Data Validation
        if fips_col not in dataframe.columns:
            raise ValueError(f"FIPS column '{fips_col}' not found in DataFrame.")
        if time_col not in dataframe.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame.")

        #  Convert relevant columns to NumPy for faster slicing
        self.input_data = self.df[self.input_cols].astype(np.float32).values
        self.target_data = self.df[self.target_col].astype(np.float32).values.reshape(-1, 1) # Ensure 2D
        self.temporal_data = self.df[self.temporal_cols].astype(np.int16).values
        self.event_data = self.df[self.event_cols].astype(np.float32).values

        self.indices = self._create_sequence_indices()

        print(f"Created dataset with {len(self.indices)} samples.")

    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """
        Generates a list of valid (start_index, end_index) tuples for sequences
        across all counties. Ensures sequences do not cross county boundaries.
        """
        indices = []
        total_len = len(self.df)
        seq_len = self.context_window + self.horizon

        # Group by county to get start/end row index for each county
        county_groups = self.df.groupby(self.fips_col).indices

        print(f"Processing {len(county_groups)} counties...")
        num_skipped = 0
        #skipped_fips = []
        with tqdm(total=len(county_groups), desc="Processing counties") as pbar:
          for fips, group_indices in county_groups.items():
              start_row = group_indices[0]
              end_row = group_indices[-1]
              county_len = len(group_indices)

              if county_len >= seq_len:
                  # Iterate within the county's range in the main DataFrame
                  for i in range(county_len - seq_len + 1):
                      # Absolute start index in the main DataFrame
                      abs_start_idx = start_row + i
                      indices.append(abs_start_idx) # Store only the start index
              else:
                  #print warning for counties too short
                  #print(f"Warning: County {fips} has length {county_len}, "
                  #f"which is less than context+horizon ({seq_len}). Skipping.")
                  num_skipped += 1
                  #skipped_fips.append(fips)
                  pass

              pbar.update(1)

        print("Finished creating indices.")
        print(f"Skipped {num_skipped}")
        return indices

    def __len__(self) -> int:
        """Returns the total number of valid sequences across all counties."""
        return len(self.indices)

    def _process_temporal_features(self, temporal_slice: np.ndarray) -> torch.Tensor:
        """
        Processes temporal features for a given sequence window.
        Currently returns raw features. Could be extended for encoding (e.g., cyclical).

        Args:
            temporal_slice (np.ndarray): Raw temporal features for the context window.
                                         Shape: (context_window, num_temporal_features)

        Returns:
            torch.Tensor: Processed temporal features.
                          Shape: (context_window, num_temporal_features)
        """
        return torch.tensor(temporal_slice, dtype=torch.int16)

    def _process_event_features(self, event_slice: np.ndarray) -> torch.Tensor:
        """
        Processes event features for a given sequence window by summing counts
        over the window for each event type.

        Args:
            event_slice (np.ndarray): Raw event features for the context window.
                                      Shape: (context_window, num_event_features)

        Returns:
            torch.Tensor: Processed event features (summed counts).
                          Shape: (num_event_features,)
        """
        # Convert event_slice to a PyTorch tensor
        event_tensor = torch.tensor(event_slice, dtype=torch.float32)
        # Sum along the time dimention.
        summed_events = torch.sum(event_tensor, dim=0)

        return summed_events


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single sample (input sequence, target sequence, features).

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                'input_ts': Input time series (context_window, num_input_features)
                'target_ts': Target time series (horizon, 1)
                'temporal_features': Processed temporal feats (context_window, num_temporal_features)
                'event_features': Processed event features (num_event_features,)
                'frequency': The provided frequency value (scalar tensor) - Optional
        """
        # 1. Get the start index of the sequence in the original DataFrame
        start_idx = self.indices[idx]

        # 2. Define slice boundaries
        input_end_idx = start_idx + self.context_window
        target_end_idx = input_end_idx + self.horizon

        # 3. Slice the pre-converted NumPy arrays
        input_ts_slice = self.input_data[start_idx:input_end_idx, :]
        target_ts_slice = self.target_data[input_end_idx:target_end_idx, :]
        temporal_features_slice = self.temporal_data[start_idx:input_end_idx, :] # Temporal features aligned with input
        event_features_slice = self.event_data[start_idx:input_end_idx, :]       # Event features aligned with input

        # Verify that the sequence belongs to a single county
        fips_in_sequence = self.df.iloc[start_idx:target_end_idx][self.fips_col].nunique()
        if fips_in_sequence > 1:
            print(f"Warning: Sequence at index {idx} (starts at df row {start_idx}) "
                    f"spans multiple fips codes! Check sorting and _create_sequence_indices.")

        # 4. Process features
        processed_temporal = self._process_temporal_features(temporal_features_slice)
        processed_events = self._process_event_features(event_features_slice)

        # 5. Convert slices to Tensors
        input_ts = torch.tensor(input_ts_slice, dtype=torch.float32)
        input_padding = torch.zeros_like(input_ts)
        target_ts = torch.tensor(target_ts_slice, dtype=torch.float32)
        frequency = torch.tensor(self.frequency, dtype=torch.long)
        # 6. Prepare output dictionary
        sample = {
            'input_ts': input_ts,             # Shape: (context_window, num_input_features)
            'input_padding': input_padding,   # Shape: (context_window, num_input_features)
            'target_ts': target_ts,           # Shape: (horizon, 1)
            'temporal_features': processed_temporal, # Shape: (context_window, num_temporal_features)
            'event_features': processed_events,      # Shape: (num_event_features,)
            'frequency': frequency           
        }
        return sample