import torch
import torch.nn as nn
from typing import Tuple, Dict


class StatePredictor(nn.Module):
    """Predicts next vehicle state given sequence of controls and current state"""

    def __init__(self, state_dim: int, control_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim

        # LSTM for processing control sequence
        self.control_lstm = nn.LSTM(
            input_size=control_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # MLP for combining current state and processed control sequence
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, current_state: torch.Tensor, control_sequence: torch.Tensor) -> torch.Tensor:
        # Process control sequence
        control_features, _ = self.control_lstm(control_sequence)
        control_features = control_features[:, -1, :]  # Take last hidden state

        # Combine with current state
        combined = torch.cat([current_state, control_features], dim=1)

        # Predict next state
        next_state = self.predictor(combined)
        return next_state


class ControlPredictor(nn.Module):
    """Predicts sequence of controls to achieve desired state"""

    def __init__(self, state_dim: int, control_dim: int, sequence_length: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.sequence_length = sequence_length

        # Encoder for processing current and desired states
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Decoder for generating control sequence
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, control_dim)

    def forward(self, current_state: torch.Tensor, desired_state: torch.Tensor) -> torch.Tensor:
        # Encode states
        states = torch.cat([current_state, desired_state], dim=1)
        encoded = self.state_encoder(states)

        # Prepare decoder input (repeat encoded state for sequence generation)
        decoder_input = encoded.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Generate control sequence
        outputs, _ = self.decoder(decoder_input)
        control_sequence = self.output_proj(outputs)

        return control_sequence