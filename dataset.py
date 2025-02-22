import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional


def read_data(file_path: str, columns: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    """
    Read CSV data into a pandas DataFrame with timestamp parsing

    Args:
        file_path (str): Path to the CSV file
        columns (Optional[list[str]]): List of specific columns to read. If None, reads all columns

    Returns:
        Optional[pd.DataFrame]: DataFrame containing the requested data, None if file not found
    """
    try:
        # Read CSV with timestamp parsing
        df = pd.read_csv(file_path)

        # Parse timestamp column if it exists
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'])

        # Return specific columns if requested
        if columns is not None:
            available_columns = [col for col in columns if col in df.columns]
            if not available_columns:
                return None
            return df[['ts'] + available_columns] if 'ts' in df.columns else df[available_columns]

        return df

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def _fit_scaler(data: pd.DataFrame) -> Dict:
    """Compute mean and std for normalization"""
    return {
        'mean': data.mean(),
        'std': data.std()
    }


def _normalize(data: np.ndarray, scaler: Dict) -> np.ndarray:
    """Normalize data using pre-computed statistics"""
    return (data - scaler['mean'].values) / scaler['std'].values


class VehicleDataset(Dataset):
    def __init__(self, ahrs_df: pd.DataFrame, control_df: pd.DataFrame,
                 sequence_length: int, prediction_horizon: int, mode: str = 'state',
                 sample_rate: int = 50, max_sequences: int = 5000):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode

        # Clean data first
        ahrs_df = (ahrs_df.interpolate(method='linear')
                   .bfill()
                   .ffill())
        control_df = (control_df.interpolate(method='linear')
                      .bfill()
                      .ffill())

        # Subsample the dataframes
        ahrs_df = ahrs_df.iloc[::sample_rate].reset_index(drop=True)
        control_df = control_df.iloc[::sample_rate].reset_index(drop=True)

        # Merge dataframes on timestamp
        self.merged_df = pd.merge_asof(ahrs_df, control_df, on='ts')

        # Remove any remaining NaN values after merge
        self.merged_df = self.merged_df.dropna()

        # Store column names
        self.state_columns = [col for col in ahrs_df.columns if col != 'ts']
        self.control_columns = [col for col in control_df.columns if col != 'ts']

        # Compute normalization parameters
        self.state_scaler = _fit_scaler(self.merged_df[self.state_columns])
        self.control_scaler = _fit_scaler(self.merged_df[self.control_columns])

        # Create sequence indices
        self.valid_indices = self._get_valid_indices()

        # Limit the number of sequences if necessary
        if max_sequences and len(self.valid_indices) > max_sequences:
            self.valid_indices = self.valid_indices[:max_sequences]

        print(f"Dataset created with {len(self.valid_indices)} sequences")
        print(f"Data shape after preprocessing: {self.merged_df.shape}")

    def _get_valid_indices(self) -> List[int]:
        """Get valid starting indices for sequences"""
        if self.mode == 'state':
            return list(range(len(self.merged_df) - self.sequence_length - self.prediction_horizon))
        else:  # control prediction
            return list(range(len(self.merged_df) - self.prediction_horizon))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start_idx = self.valid_indices[idx]

        if self.mode == 'state':
            # Get control sequence
            control_seq = self.merged_df[self.control_columns].iloc[
                          start_idx:start_idx + self.sequence_length].values

            # Get current and next state
            current_state = self.merged_df[self.state_columns].iloc[
                start_idx + self.sequence_length - 1].values
            next_state = self.merged_df[self.state_columns].iloc[
                start_idx + self.sequence_length].values

            return (
                torch.FloatTensor(_normalize(control_seq, self.control_scaler)),
                torch.FloatTensor(_normalize(current_state, self.state_scaler)),
                torch.FloatTensor(_normalize(next_state, self.state_scaler))
            )
        else:
            # Get current and future state
            current_state = self.merged_df[self.state_columns].iloc[start_idx].values
            future_state = self.merged_df[self.state_columns].iloc[
                start_idx + self.prediction_horizon].values

            # Get control sequence
            control_seq = self.merged_df[self.control_columns].iloc[
                          start_idx:start_idx + self.prediction_horizon].values

            return (
                torch.FloatTensor(_normalize(current_state, self.state_scaler)),
                torch.FloatTensor(_normalize(future_state, self.state_scaler)),
                torch.FloatTensor(_normalize(control_seq, self.control_scaler))
            )
