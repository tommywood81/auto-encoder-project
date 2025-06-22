# Data Pipeline Documentation

## Overview

The fraud detection pipeline processes raw transaction and identity data through four main stages:

1. **Data Cleaning** (`clean`) - Cleans and preprocesses raw data
2. **Feature Engineering** (`engineer`) - Creates new features from cleaned data  
3. **Data Processing** (`process`) - Prepares final data for modeling
4. **Model Training** (`train`) - Trains autoencoder model and evaluates performance

## Quick Start

### Run Complete Pipeline (Data + Training)
```bash
python run_pipeline.py
```

### Run Data Processing Only (No Training)
```bash
python run_pipeline.py --stages clean engineer process
```

### Run Only Model Training (Skip Data Processing)
```bash
python run_pipeline.py --stages train
```

### Run Specific Stages
```bash
# Run only data cleaning
python run_pipeline.py --stages clean

# Run only feature engineering
python run_pipeline.py --stages engineer

# Run only data processing
python run_pipeline.py --stages process

# Run only model training
python run_pipeline.py --stages train

# Run multiple stages
python run_pipeline.py --stages clean engineer process
```

### Force Rerun (Overwrite Existing Output)
```bash
python run_pipeline.py --force-rerun
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--stages` | Pipeline stages to run: `clean`, `engineer`, `process`, `train`, `all` | `all` |
| `--force-rerun` | Force rerun all stages even if output exists | `False` |
| `--log-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--log-file` | Save logs to file (optional) | `None` |

## Data Directory Structure

```
data/
├── raw/                    # Original downloaded data
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
├── cleaned/                # After data cleaning
│   ├── train_cleaned.csv
│   ├── cleaning_stats.json
│   └── label_encoders.json
├── engineered/             # After feature engineering
│   ├── train_features.csv
│   └── feature_info.json
├── processed/              # Final processed data for modeling
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_names.txt
└── intermediate/           # Temporary files

models/                     # Trained models
└── autoencoder_fraud_detection.pth

results/                    # Training results and metrics
├── metrics.json
├── training_losses.npy
├── pipeline_report.json
└── evaluation_plots/
```

## Pipeline Stages

### Stage 1: Data Cleaning (`clean`)

**Input**: Raw transaction and identity CSV files  
**Output**: Cleaned CSV file with statistics and encoders

**Operations**:
- Merge transaction and identity data
- Remove unnecessary columns (TransactionID, TransactionDT, etc.)
- Remove columns with >50% missing values
- Handle outliers using IQR method
- Encode categorical variables
- Fill missing values

**Files Generated**:
- `data/cleaned/train_cleaned.csv` - Cleaned dataset
- `data/cleaned/cleaning_stats.json` - Cleaning statistics
- `data/cleaned/label_encoders.json` - Label encoder mappings

### Stage 2: Feature Engineering (`engineer`)

**Input**: Cleaned CSV file  
**Output**: Engineered CSV file with new features

**New Features Created**:
- **Transaction Features** (5 features):
  - `amount_log` - Log transformation of transaction amount
  - `amount_sqrt` - Square root transformation of transaction amount
  - `card_count` - Number of card-related fields filled
  - `email_count` - Number of email-related fields filled
  - `addr_count` - Number of address-related fields filled

- **Identity Features** (5 features):
  - `v_features_count` - Count of V features filled
  - `v_features_mean` - Mean of V features
  - `v_features_std` - Standard deviation of V features
  - `v_features_sum` - Sum of V features
  - `id_features_count` - Count of ID features filled

- **Interaction Features** (3 features):
  - `amount_card_interaction` - Transaction amount × card count
  - `amount_email_interaction` - Transaction amount × email count
  - `v_count_mean_interaction` - V features count × V features mean

- **Statistical Features** (4 features):
  - `v_features_q25` - 25th percentile of V features
  - `v_features_q75` - 75th percentile of V features
  - `v_features_iqr` - Interquartile range of V features
  - `v_features_range` - Range of V features

**Files Generated**:
- `data/engineered/train_features.csv` - Dataset with engineered features
- `data/engineered/feature_info.json` - Feature information and descriptions

### Stage 3: Data Processing (`process`)

**Input**: Engineered CSV file  
**Output**: Processed NumPy arrays and preprocessing objects

**Operations**:
- Split data into train/test sets (80/20)
- Scale features using StandardScaler
- Prepare autoencoder training data (non-fraudulent only)
- Save processed data for fast loading

**Files Generated**:
- `data/processed/X_train.npy` - Training features
- `data/processed/X_test.npy` - Test features
- `data/processed/y_train.npy` - Training labels
- `data/processed/y_test.npy` - Test labels
- `data/processed/scaler.pkl` - Fitted StandardScaler
- `data/processed/label_encoders.pkl` - Fitted label encoders
- `data/processed/feature_names.txt` - Feature names list

### Stage 4: Model Training (`train`)

**Input**: Processed NumPy arrays  
**Output**: Trained model and evaluation results

**Operations**:
- Initialize autoencoder with architecture [input_dim → 64 → 32 → 64 → input_dim]
- Train on non-fraudulent data only (anomaly detection approach)
- Detect anomalies using reconstruction error threshold
- Evaluate model performance with comprehensive metrics
- Save trained model and results

**Files Generated**:
- `models/autoencoder_fraud_detection.pth` - Trained model with metadata
- `results/metrics.json` - Model performance metrics
- `results/training_losses.npy` - Training loss history
- `results/evaluation_plots/` - Performance visualization plots

## Data Statistics

| Stage | Rows | Features | Size | Description |
|-------|------|----------|------|-------------|
| Raw | 590,540 | 434 | 1.3 GB | Original transaction + identity data |
| Cleaned | 590,540 | 217 | 500 MB | After cleaning and preprocessing |
| Engineered | 590,540 | 233 | 580 MB | After feature engineering |
| Processed | 472,432 train, 118,108 test | 232 | 1.0 GB | Final processed data |

**Fraud Distribution**:
- **Total transactions**: 590,540
- **Fraudulent transactions**: 20,663 (3.5%)
- **Legitimate transactions**: 569,877 (96.5%)

## Usage Examples

### Complete Workflow (Recommended)
```bash
# Run complete pipeline from data to trained model
python run_pipeline.py

# Run with detailed logging
python run_pipeline.py --log-level DEBUG --log-file pipeline.log
```

### Development Workflow
```bash
# First time setup - run complete pipeline
python run_pipeline.py

# Modify feature engineering, then rerun
python run_pipeline.py --stages engineer --force-rerun

# Test only model training
python run_pipeline.py --stages train

# Retrain model with different parameters
python run_pipeline.py --stages train --force-rerun
```

### Production Workflow
```bash
# Run with detailed logging and file output
python run_pipeline.py --log-level DEBUG --log-file production.log

# Run specific stages with force rerun
python run_pipeline.py --stages clean engineer --force-rerun
```

### Monitoring and Debugging
```bash
# Check pipeline status
python run_pipeline.py --stages process

# View generated report
cat results/pipeline_report.json

# Check model performance
cat results/metrics.json
```

## Model Architecture

The autoencoder uses the following architecture:
- **Input Layer**: 232 features (after processing)
- **Encoder**: 232 → 64 → 32 (compression)
- **Decoder**: 32 → 64 → 232 (reconstruction)
- **Activation**: ReLU for hidden layers, linear for output
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 1e-3
- **Training**: 20 epochs, batch size 256

## Output Files

### Pipeline Report
The pipeline generates a comprehensive report at `results/pipeline_report.json` containing:
- Timestamp of execution
- Status of each pipeline stage
- File counts and sizes
- Execution statistics

### Model Results
- **`results/metrics.json`**: Model performance metrics (accuracy, precision, recall, F1-score)
- **`results/training_losses.npy`**: Training loss history for plotting
- **`results/evaluation_plots/`**: Performance visualization plots
- **`models/autoencoder_fraud_detection.pth`**: Trained model with all metadata

### Logging
- Console output with progress indicators
- Optional log file output
- Detailed error messages and stack traces

## Error Handling

The pipeline includes robust error handling:
- **Prerequisites check**: Validates raw data exists
- **Stage dependencies**: Ensures previous stages completed
- **File existence checks**: Skips stages if output exists (unless `--force-rerun`)
- **Exception handling**: Graceful failure with detailed error messages
- **Keyboard interrupt**: Clean shutdown on Ctrl+C
- **GPU/CPU fallback**: Automatically uses CPU if CUDA unavailable

## Performance

Typical execution times (on modern hardware):
- **Data Cleaning**: ~30-60 seconds
- **Feature Engineering**: ~60-120 seconds  
- **Data Processing**: ~10-30 seconds
- **Model Training**: ~2-5 minutes (CPU) / ~30-60 seconds (GPU)
- **Total Pipeline**: ~5-10 minutes

## Model Inference

After training, the model can be used for inference:

```python
import torch
from src.autoencoder import Autoencoder

# Load trained model
checkpoint = torch.load('models/autoencoder_fraud_detection.pth')
model = Autoencoder(input_dim=232, hidden_dims=[64, 32])
model.load_state_dict(checkpoint['model_state_dict'])

# Load preprocessing objects
scaler = checkpoint['scaler']
threshold = checkpoint['threshold']

# Preprocess new data and make predictions
# (Use the same preprocessing pipeline)
```

## Troubleshooting

### Common Issues

1. **Missing raw data**: Ensure `data/raw/` contains the required CSV files
2. **Memory errors**: Consider processing smaller chunks or increasing system memory
3. **Permission errors**: Check file/directory permissions
4. **CUDA errors**: Model will automatically fall back to CPU
5. **JSON serialization errors**: Fixed in current version

### Debug Mode
```bash
python run_pipeline.py --log-level DEBUG
```

### Clean Restart
```bash
# Remove all processed data and restart
rm -rf data/cleaned data/engineered data/processed models results
python run_pipeline.py
```

### Model Retraining
```bash
# Retrain model only
python run_pipeline.py --stages train --force-rerun
``` 