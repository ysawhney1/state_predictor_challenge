import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import pandas as pd


def evaluate_model(model: torch.nn.Module,
                   test_loader: DataLoader,
                   scaler: Dict,
                   mode: str = 'state') -> Tuple[Dict, List[float], List[float]]:
    """
    Evaluate model performance with multiple metrics
    """
    model.eval()
    metrics = {
        'mse': [],  # Mean Squared Error
        'mae': [],  # Mean Absolute Error
        'rmse': [],  # Root Mean Squared Error
        'mape': [],  # Mean Absolute Percentage Error
        'r2': [],  # R-squared score
        'max_error': [],  # Maximum Error
        'std_error': []  # Standard Deviation of Error
    }

    predictions = []
    true_values = []

    with torch.no_grad():
        for batch in test_loader:
            if mode == 'state':
                control_seq, current_state, next_state = batch
                pred = model(current_state, control_seq)
                true = next_state
            else:
                current_state, desired_state, true_controls = batch
                pred = model(current_state, desired_state)
                true = true_controls

            # Denormalize predictions and true values
            pred_denorm = pred.numpy() * scaler['std'].to_numpy() + scaler['mean'].to_numpy()
            true_denorm = true.numpy() * scaler['std'].to_numpy() + scaler['mean'].to_numpy()

            # Calculate errors
            errors = pred_denorm - true_denorm
            abs_errors = np.abs(errors)
            squared_errors = errors ** 2

            # Update metrics
            metrics['mse'].append(np.mean(squared_errors))
            metrics['mae'].append(np.mean(abs_errors))
            metrics['rmse'].append(np.sqrt(np.mean(squared_errors)))

            # Calculate MAPE avoiding division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs(errors / true_denorm)) * 100
                metrics['mape'].append(np.nan_to_num(mape))

            # Calculate R-squared
            ss_res = np.sum(squared_errors)
            ss_tot = np.sum((true_denorm - np.mean(true_denorm)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            metrics['r2'].append(r2)

            metrics['max_error'].append(np.max(abs_errors))
            metrics['std_error'].append(np.std(errors))

            predictions.extend(pred_denorm)
            true_values.extend(true_denorm)

    # Calculate average metrics
    final_metrics = {
        key: np.mean(values) for key, values in metrics.items()
    }

    # Add feature-wise metrics
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    feature_metrics = {}

    for i in range(predictions.shape[1]):
        feature_metrics[f'feature_{i}'] = {
            'mse': np.mean((predictions[:, i] - true_values[:, i]) ** 2),
            'mae': np.mean(np.abs(predictions[:, i] - true_values[:, i])),
            'correlation': np.corrcoef(predictions[:, i], true_values[:, i])[0, 1]
        }

    final_metrics['feature_wise'] = feature_metrics

    return final_metrics, predictions, true_values


def save_metrics_to_file(state_metrics: Dict, control_metrics: Dict,
                         ahrs_features: List[str], control_features: List[str],
                         save_dir: str = 'prediction_stats'):
    """
    Save all metrics to a formatted text file
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(save_dir, f'model_metrics_{timestamp}.txt')

    with open(filepath, 'w') as f:
        # Write header
        f.write("=" * 50 + "\n")
        f.write("Vehicle State and Control Prediction Metrics\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        # Write State Predictor metrics
        f.write("STATE PREDICTOR METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall MSE: {state_metrics['mse']:.4f}\n")
        f.write(f"Overall RMSE: {state_metrics['rmse']:.4f}\n")
        f.write(f"Overall MAE: {state_metrics['mae']:.4f}\n")
        f.write(f"Overall MAPE: {state_metrics['mape']:.2f}%\n")
        f.write(f"Overall R-squared: {state_metrics['r2']:.4f}\n")
        f.write(f"Maximum Error: {state_metrics['max_error']:.4f}\n")
        f.write(f"Standard Deviation of Error: {state_metrics['std_error']:.4f}\n\n")

        f.write("Feature-wise State Metrics:\n")
        for i, feature in enumerate(ahrs_features):
            metrics_i = state_metrics['feature_wise'][f'feature_{i}']
            f.write(f"\n{feature}:\n")
            f.write(f"  MSE: {metrics_i['mse']:.4f}\n")
            f.write(f"  MAE: {metrics_i['mae']:.4f}\n")
            f.write(f"  Correlation: {metrics_i['correlation']:.4f}\n")

        # Write Control Predictor metrics
        f.write("\n\nCONTROL PREDICTOR METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall MSE: {control_metrics['mse']:.4f}\n")
        f.write(f"Overall RMSE: {control_metrics['rmse']:.4f}\n")
        f.write(f"Overall MAE: {control_metrics['mae']:.4f}\n")
        f.write(f"Overall MAPE: {control_metrics['mape']:.2f}%\n")
        f.write(f"Overall R-squared: {control_metrics['r2']:.4f}\n")
        f.write(f"Maximum Error: {control_metrics['max_error']:.4f}\n")
        f.write(f"Standard Deviation of Error: {control_metrics['std_error']:.4f}\n\n")

        f.write("Feature-wise Control Metrics:\n")
        for i, feature in enumerate(control_features):
            metrics_i = control_metrics['feature_wise'][f'feature_{i}']
            f.write(f"\n{feature}:\n")
            f.write(f"  MSE: {metrics_i['mse']:.4f}\n")
            f.write(f"  MAE: {metrics_i['mae']:.4f}\n")
            f.write(f"  Correlation: {metrics_i['correlation']:.4f}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("End of Metrics Report\n")

    print(f"\nMetrics saved to: {filepath}")


def plot_grouped_predictions(predictions: List[float],
                             true_values: List[float],
                             feature_names: List[str],
                             save_dir: str = 'plots'):
    """
    Plot predictions grouped by measurement type and save as PNGs
    """
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define groups and their features
    feature_groups = {
        'angles': ['roll_deg', 'pitch_deg', 'yaw_deg'],
        'accelerations': ['ax_mps2', 'ay_mps2', 'az_mps2'],
        'angular_rates': ['omega_x_dps', 'omega_y_dps', 'omega_z_dps'],
        'velocities': ['ve_mps', 'vn_mps', 'vu_mps']
    }

    # Define feature descriptions
    feature_descriptions = {
        'roll_deg': 'Roll Angle (degrees)',
        'pitch_deg': 'Pitch Angle (degrees)',
        'yaw_deg': 'Yaw Angle (degrees)',
        'ax_mps2': 'X Acceleration (m/s²)',
        'ay_mps2': 'Y Acceleration (m/s²)',
        'az_mps2': 'Z Acceleration (m/s²)',
        'omega_x_dps': 'X Angular Rate (deg/s)',
        'omega_y_dps': 'Y Angular Rate (deg/s)',
        'omega_z_dps': 'Z Angular Rate (deg/s)',
        've_mps': 'East Velocity (m/s)',
        'vn_mps': 'North Velocity (m/s)',
        'vu_mps': 'Up Velocity (m/s)'
    }

    # Plot each group
    for group_name, group_features in feature_groups.items():
        # Get indices of features in this group
        feature_indices = [feature_names.index(feat) for feat in group_features]

        # Create subplot for this group
        fig, axes = plt.subplots(len(group_features), 1, figsize=(15, 5 * len(group_features)))
        axes = np.atleast_1d(axes)

        for i, (feat_idx, feat_name) in enumerate(zip(feature_indices, group_features)):
            ax = axes[i]

            # Plot with different colors and styles
            ax.plot(true_values[:100, feat_idx], 'b-', label='True', alpha=0.7, linewidth=2)
            ax.plot(predictions[:100, feat_idx], 'r--', label='Predicted', alpha=0.7, linewidth=2)

            ax.set_title(f'{feature_descriptions[feat_name]}', fontsize=12, pad=10)
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add some padding to the y-axis
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        plt.suptitle(f'State Predictions - {group_name.replace("_", " ").title()}',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        # Save the plot
        filename = f"state_predictions_{group_name}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved as: {save_path}")


def plot_control_predictions(predictions: List[float],
                             true_values: List[float],
                             feature_names: List[str],
                             save_dir: str = 'plots'):
    """
    Plot control predictions
    """
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define feature descriptions
    feature_descriptions = {
        'gear': 'Gear Setting (0:neutral, 1:forward, 2:reverse)',
        'throttle': 'Throttle Value (0.0-1.0)',
        'trim': 'Trim Setting (0-2)',
        'turn': 'Turn Value (-1.0:left to 1.0:right)'
    }

    # Create plot
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(15, 5 * len(feature_names)))
    axes = np.atleast_1d(axes)

    for i, feat_name in enumerate(feature_names):
        ax = axes[i]

        ax.plot(true_values[:100, i], 'b-', label='True', alpha=0.7, linewidth=2)
        ax.plot(predictions[:100, i], 'r--', label='Predicted', alpha=0.7, linewidth=2)

        ax.set_title(f'{feature_descriptions[feat_name]}', fontsize=12, pad=10)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.suptitle('Control Predictions', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save the plot
    filename = "control_predictions.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as: {save_path}")
