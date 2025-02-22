import os
from config import (AHRS_COLUMNS, CONTROL_COLUMNS, AHRS_FILE, CONTROL_FILE,
                    STATE_PREDICTOR_CONFIG, CONTROL_PREDICTOR_CONFIG, DATA_CONFIG)
from models import StatePredictor, ControlPredictor
from dataset import VehicleDataset, read_data
import torch
from torch.utils.data import DataLoader, random_split
from evaluate import evaluate_model, plot_control_predictions, plot_grouped_predictions, save_metrics_to_file


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            try:
                if isinstance(model, StatePredictor):
                    control_seq, current_state, next_state = batch
                    pred = model(current_state, control_seq)
                    loss = criterion(pred, next_state)
                else:  # ControlPredictor
                    current_state, desired_state, true_controls = batch
                    pred = model(current_state, desired_state)
                    loss = criterion(pred, true_controls)

                # Check for NaN loss
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                print(f"Error in batch: {e}")
                continue

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
        else:
            print(f'Epoch {epoch + 1}, No valid batches')


if __name__ == "__main__":
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Read data frames
    ahrs_df = read_data(os.path.join(current_dir, AHRS_FILE), list(AHRS_COLUMNS.keys()))
    control_df = read_data(os.path.join(current_dir, CONTROL_FILE), list(CONTROL_COLUMNS.keys()))

    if ahrs_df is not None and control_df is not None:
        print(f"Original data shapes - AHRS: {ahrs_df.shape}, Control: {control_df.shape}")

        # Create datasets with sampling
        state_dataset = VehicleDataset(
            ahrs_df,
            control_df,
            STATE_PREDICTOR_CONFIG['sequence_length'],
            STATE_PREDICTOR_CONFIG['prediction_horizon'],
            mode='state',
            sample_rate=DATA_CONFIG['sample_rate'],
            max_sequences=DATA_CONFIG['max_sequences']
        )

        control_dataset = VehicleDataset(
            ahrs_df,
            control_df,
            CONTROL_PREDICTOR_CONFIG['sequence_length'],
            CONTROL_PREDICTOR_CONFIG['prediction_horizon'],
            mode='control',
            sample_rate=DATA_CONFIG['sample_rate'],
            max_sequences=DATA_CONFIG['max_sequences']
        )

        # Calculate split sizes based on actual dataset size
        state_train_size = int(len(state_dataset) * DATA_CONFIG['train_split'])
        state_test_size = len(state_dataset) - state_train_size

        control_train_size = int(len(control_dataset) * DATA_CONFIG['train_split'])
        control_test_size = len(control_dataset) - control_train_size

        print(f"\nSplitting datasets:")
        print(f"State dataset - Total: {len(state_dataset)}, Train: {state_train_size}, Test: {state_test_size}")
        print(
            f"Control dataset - Total: {len(control_dataset)}, Train: {control_train_size}, Test: {control_test_size}")

        # Split datasets into train and test
        state_train_dataset, state_test_dataset = random_split(
            state_dataset, [state_train_size, state_test_size])
        control_train_dataset, control_test_dataset = random_split(
            control_dataset, [control_train_size, control_test_size])

        print(f"Number of training sequences - State: {len(state_dataset)}, Control: {len(control_dataset)}")

        # Create dataloaders
        state_train_loader = DataLoader(
            state_train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        state_test_loader = DataLoader(
            state_test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )

        control_train_loader = DataLoader(
            control_train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        control_test_loader = DataLoader(
            control_test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )

        # Initialize models
        state_predictor = StatePredictor(
            state_dim=len(AHRS_COLUMNS),
            control_dim=len(CONTROL_COLUMNS)
        )

        control_predictor = ControlPredictor(
            state_dim=len(AHRS_COLUMNS),
            control_dim=len(CONTROL_COLUMNS),
            sequence_length=CONTROL_PREDICTOR_CONFIG['prediction_horizon']
        )

        # Training setup
        criterion = torch.nn.MSELoss()
        state_optimizer = torch.optim.Adam(
            state_predictor.parameters(),
            lr=STATE_PREDICTOR_CONFIG['learning_rate']
        )
        control_optimizer = torch.optim.Adam(
            control_predictor.parameters(),
            lr=CONTROL_PREDICTOR_CONFIG['learning_rate']
        )

        # Train models
        print("\nTraining State Predictor...")
        train_model(state_predictor, state_train_loader, criterion, state_optimizer)

        print("\nTraining Control Predictor...")
        train_model(control_predictor, control_train_loader, criterion, control_optimizer)

        # Evaluate models
        print("\nEvaluating State Predictor...")
        state_metrics, state_preds, state_true = evaluate_model(
            state_predictor,
            state_test_loader,
            state_dataset.state_scaler,
            mode='state'
        )

        print("\nEvaluating Control Predictor...")
        control_metrics, control_preds, control_true = evaluate_model(
            control_predictor,
            control_test_loader,
            control_dataset.control_scaler,
            mode='control'
        )

        # Save metrics to file
        save_metrics_to_file(
            state_metrics,
            control_metrics,
            list(AHRS_COLUMNS.keys())[1:],  # Skip 'ts' column
            list(CONTROL_COLUMNS.keys())[1:],  # Skip 'ts' column
            save_dir='prediction_stats'
        )

        # Plot results
        print("\nPlotting State Predictions...")
        state_features = [
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'ax_mps2', 'ay_mps2', 'az_mps2',
            'omega_x_dps', 'omega_y_dps', 'omega_z_dps',
            've_mps', 'vn_mps', 'vu_mps'
        ]
        plot_grouped_predictions(
            state_preds,
            state_true,
            state_features,
            save_dir='prediction_plots'
        )

        print("\nPlotting Control Predictions...")
        control_features = ['gear', 'throttle', 'trim', 'turn']
        plot_control_predictions(
            control_preds,
            control_true,
            control_features,
            save_dir='prediction_plots'
        )

        # Save models
        torch.save(state_predictor.state_dict(), 'state_predictor.pth')
        torch.save(control_predictor.state_dict(), 'control_predictor.pth')
        print("\nModels saved successfully")

    else:
        print("Error loading data files")
