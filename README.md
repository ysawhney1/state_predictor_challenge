# Vehicle State and Control Predictor

This project implements two neural network predictors for vehicle state and control:
1. **State Predictor**: Predicts the next vehicle state given a sequence of vehicle controls and current state
2. **Control Predictor**: Predicts the sequence of vehicle controls needed to achieve a desired vehicle state

## Model Architecture

### State Predictor
- Combines processed controls with current state using MLP
- Predicts next vehicle state
- Designed to capture complex dependencies in vehicle dynamics

### Control Predictor
- Encoder-decoder architecture
- Processes current and desired states
- Generates sequence of control commands
- Optimized for generating feasible control sequences

## Project Structure
```
state_predictor_challenge/
├── config.py          # Configuration parameters
├── dataset.py         # Data loading and preprocessing
├── models.py          # Neural network model definitions
├── evaluate.py        # Evaluation metrics and visualization
├── main.py           # Main training and evaluation script
├── prediction_plots/  # Generated visualization plots
└── prediction_stats/  # Performance metrics and statistics
```

## How to Run

### 1. Setup Development Environment

First, ensure you have Python 3.8+ installed. Then set up your development environment:

```bash
# Clone the repository
git clone [your-repo-url]
cd state_predictor_challenge

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows:
venv\Scripts\activate
# For Linux/Mac:
source venv/bin/activate

# Install required packages
pip install torch pandas numpy matplotlib
```

### 2. Prepare Data Files

Place your data files in the project root directory:
- `ahrs.csv`: Vehicle state measurements
- `vehicle_control.csv`: Control inputs

### 3. Project Structure
Ensure your project has the following structure:
```
state_predictor_challenge/
├── config.py
├── dataset.py
├── models.py
├── evaluate.py
├── main.py
├── README.md
├── ahrs.csv
└── vehicle_control.csv
```

### 4. Configure Parameters (Optional)

You can adjust the model parameters in `config.py`:
```python
# Adjust data processing parameters
DATA_CONFIG = {
    'sample_rate': 10,        # Increase to reduce data size
    'train_split': 0.8,       # Training/testing split
    'max_sequences': 10000    # Maximum sequences to process
}

# Adjust model parameters
STATE_PREDICTOR_CONFIG = {
    'sequence_length': 10,
    'batch_size': 32,
    'learning_rate': 0.001
}
```

### 5. Run the Training

Execute the main script:
```bash
python main.py
```

The script will:
1. Load and preprocess the data files
2. Create and train both predictors
3. Generate evaluation metrics
4. Create visualization plots
5. Save the trained models

### 6. Check Outputs

After running, check the following directories for results:

#### Generated Plots (`prediction_plots/`)
- `state_predictions_angles.png`
  - Shows roll, pitch, and yaw predictions
- `state_predictions_accelerations.png`
  - Shows X, Y, Z acceleration predictions
- `state_predictions_angular_rates.png`
  - Shows angular rate predictions
- `state_predictions_velocities.png`
  - Shows velocity predictions in ENU frame
- `control_predictions.png`
  - Shows control variable predictions

#### Performance Metrics (`prediction_stats/`)
- `model_metrics_[timestamp].txt`
  - Contains MSE, RMSE, MAE, etc.
  - Feature-wise performance analysis
  - Correlation metrics

#### Saved Models (root directory)
- `state_predictor.pth`
- `control_predictor.pth`

### 7. Troubleshooting

Common issues and solutions:

1. Memory Error:
   - Increase `sample_rate` in `config.py`
   - Decrease `max_sequences` in `config.py`
   - Reduce `batch_size` in predictor configs

2. Data Loading Error:
   - Ensure CSV files are in the correct location
   - Check CSV file format matches expected structure

3. CUDA Out of Memory:
   - Reduce batch size
   - Use CPU by removing CUDA calls

### 8. Expected Output

When running successfully, you should see console output like:
```
Original data shapes - AHRS: (345001, 13), Control: (537903, 5)
Dataset created with 5000 sequences
Data shape after preprocessing: (6900, 17)
Number of training sequences - State: 5000, Control: 5000

Training State Predictor...
Epoch 1, Loss: 0.4928
...
Epoch 10, Loss: 0.3355

Training Control Predictor...
Epoch 1, Loss: 0.6366
...
Epoch 10, Loss: 0.4276

Evaluating Models...
[Evaluation metrics will be displayed]

Plots and statistics saved successfully
```

### 9. Customizing the Run

To modify the behavior:

1. Change Training Duration:
   - Adjust `num_epochs` in `main.py`

2. Adjust Model Complexity:
   - Modify network architectures in `models.py`

3. Change Data Processing:
   - Modify preprocessing in `dataset.py`

4. Customize Visualizations:
   - Adjust plotting parameters in `evaluate.py`

## Model Performance and Results

### State Predictor Performance
- Successfully captures vehicle state transitions
- Performs well on angle predictions
- Shows some limitations in capturing sudden acceleration changes
- Provides smooth predictions for velocities and angular rates

### Control Predictor Performance
- Generates feasible control sequences
- Shows good correlation with actual control inputs
- Maintains physical constraints on control values
- Effective at predicting gear and trim settings

### Visualization Analysis
The generated plots show:
- Actual vs predicted values for all state variables
- Control prediction accuracy
- Temporal evolution of predictions
- Error distributions and patterns

### Statistical Analysis
Detailed performance metrics including:
- Overall MSE, RMSE, MAE values
- Feature-wise correlation coefficients
- Error distribution statistics
- Prediction confidence measures

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib