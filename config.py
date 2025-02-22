# Configuration for data columns and file paths

# AHRS data columns and their descriptions
AHRS_COLUMNS = {
    'roll_deg': 'degrees',  # Roll angle in degrees
    'pitch_deg': 'degrees',  # Pitch angle in degrees
    'yaw_deg': 'degrees',  # Yaw angle in degrees
    'ax_mps2': 'm/s²',  # X acceleration in FRD frame
    'ay_mps2': 'm/s²',  # Y acceleration in FRD frame
    'az_mps2': 'm/s²',  # Z acceleration in FRD frame
    'omega_x_dps': 'deg/s',  # X angular rate in FRD frame
    'omega_y_dps': 'deg/s',  # Y angular rate in FRD frame
    'omega_z_dps': 'deg/s',  # Z angular rate in FRD frame
    've_mps': 'm/s',  # East velocity in ENU frame
    'vn_mps': 'm/s',  # North velocity in ENU frame
    'vu_mps': 'm/s'  # Up velocity in ENU frame
}

# Vehicle control data columns and their descriptions
CONTROL_COLUMNS = {
    'gear': 'discrete',  # Motor setting (0: neutral, 1: forward, 2: reverse)
    'throttle': 'float',  # Throttle value (0.0 to 1.0)
    'trim': 'discrete',  # Trim command (0, 1, 2)
    'turn': 'float'  # Turn value (-1.0 to 1.0)
}

# File paths
AHRS_FILE = 'ahrs.csv'
CONTROL_FILE = 'vehicle_control.csv'

# Model configurations
STATE_PREDICTOR_CONFIG = {
    'sequence_length': 10,  # Number of previous timesteps to consider
    'prediction_horizon': 1,  # Number of future timesteps to predict
    'batch_size': 32,
    'learning_rate': 0.001
}

CONTROL_PREDICTOR_CONFIG = {
    'sequence_length': 5,  # Number of previous timesteps to consider
    'prediction_horizon': 10,  # Number of future timesteps to predict
    'batch_size': 32,
    'learning_rate': 0.001
}

DATA_CONFIG = {
    'sample_rate': 10,
    'train_split': 0.8,
    'max_sequences': 10000
}
