# Atlas-free Brain Network Transformer (BrainNet)

A PyTorch implementation of a brain network transformer for MRI brain classification using 3D spatial tokenization and attention mechanisms.

## Project Overview

This project implements an Atlas-free Brain Network Transformer that:
- Processes MRI brain connectivity data from ROI features (400 ROIs × 1632 dimensions)
- Maps ROI features to 3D brain voxel space using cluster indices
- Applies spatial block tokenization and transformer attention
- Performs binary classification on brain networks

## Dataset Structure

The project expects the following directory structure:

```
project_root/
├── toy_data/                    # Raw MRI data files (.mat format)
│   ├── label.mat               # Label file (1-indexed: 1 or 2)
│   ├── s_1_feature.mat         # ROI features for subject 1
│   ├── s_1_cluster_index.mat   # 3D cluster indices for subject 1
│   ├── s_2_feature.mat
│   ├── s_2_cluster_index.mat
│   └── ... (more subject files)
├── data_split/                 # Generated train/val/test splits (created by data_prep.py)
│   ├── data.csv
│   ├── data_clean.csv
│   ├── train_df.csv
│   ├── val_df.csv
│   └── test_df.csv
├── training_runs/              # Generated training outputs (timestamped)
│   ├── run_1704672000/
│   │   ├── best_model.pth
│   │   ├── train_log.txt
│   │   └── training_curves.jpg
│   └── run_1704672001/
└── evaluation_runs/            # Generated evaluation outputs (timestamped)
    ├── run_1704672100/
    │   ├── eval_log.txt
    │   ├── confusion_matrix.jpg
    │   └── evaluation_metrics.jpg
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shourovj/brainnet-transformer.git
cd brainnet-transformer
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- torch (>=2.0)
- scipy
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm

## Setup Instructions

### Step 1: Prepare Data Files

1. Obtain the MRI dataset files (`.mat` format)
2. Create a `toy_data/` directory in the project root
3. Place all `.mat` files in this directory:
   - `label.mat` - Brain labels (required, 1-indexed: 1 or 2)
   - `s_*_feature.mat` - ROI feature matrices for each subject
   - `s_*_cluster_index.mat` - 3D cluster index matrices for each subject

Example:
```bash
mkdir -p toy_data
cp /path/to/your/data/*.mat toy_data/
```

### Step 2: Generate Train/Val/Test Splits

Run the data preparation script to:
- Load all data files
- Validate data shapes (expected: 400×1632 features)
- Remove corrupted files
- Create stratified train/val/test splits
- Save CSV files

```bash
python data_prep.py
```

**Output:**
- `data_split/data.csv` - All data with file paths
- `data_split/data_clean.csv` - Cleaned data (corrupted files removed)
- `data_split/train_df.csv` - Training set (70%)
- `data_split/val_df.csv` - Validation set (15%)
- `data_split/test_df.csv` - Test set (15%)
- `data_split/problematic_files.csv` - Corrupted/invalid files (if any)

### Step 3: Train the Model

Train using the command-line interface with customizable parameters:

#### Basic Training
```bash
python train.py
```

#### Custom Configuration
```bash
python train.py \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --early_stopping_patience 10 \
    --device cuda
```

**Available Arguments:**
- `--num_epochs` - Number of training epochs (default: 50)
- `--batch_size` - Batch size (default: 16)
- `--learning_rate` - Learning rate (default: 1e-4)
- `--weight_decay` - L2 regularization (default: 1e-3)
- `--early_stopping_patience` - Epochs before early stopping (default: 10)
- `--lr_scheduler_patience` - LR scheduler patience (default: 5)
- `--num_workers` - Data loader workers (default: 4)
- `--device` - GPU/CPU (default: auto-detect)
- `--data_dir` - Data directory path (default: ./toy_data/)
- `--train_csv` - Training CSV path (default: ./data_split/train_df.csv)
- `--val_csv` - Validation CSV path (default: ./data_split/val_df.csv)

**Output (in `training_runs/run_<timestamp>/`):**
- `best_model.pth` - Best model checkpoint
- `train_log.txt` - Training logs with epoch-by-epoch metrics
- `training_curves.jpg` - Loss and accuracy curves (high-resolution visualization)

### Step 4: Evaluate on Test Set

Evaluate the trained model on test data:

#### Basic Evaluation
```bash
python evaluate.py
```

#### Custom Configuration
```bash
python evaluate.py \
    --checkpoint_path ./training_runs/run_1704672000/best_model.pth \
    --batch_size 32 \
    --device cuda
```

**Available Arguments:**
- `--checkpoint_path` - Path to model checkpoint (default: best_model.pth)
- `--batch_size` - Batch size (default: 16)
- `--num_workers` - Data loader workers (default: 4)
- `--device` - GPU/CPU (default: auto-detect)
- `--data_dir` - Data directory path (default: ./toy_data/)
- `--test_csv` - Test CSV path (default: ./data_split/test_df.csv)

**Output (in `evaluation_runs/run_<timestamp>/`):**
- `eval_log.txt` - Evaluation logs with detailed metrics
- `confusion_matrix.jpg` - Confusion matrix heatmap
- `evaluation_metrics.jpg` - Loss and accuracy bar charts

## Model Architecture

### Overview
The BrainNet model processes brain MRI data through these steps:

1. **ROI Connectivity Projection**: Linear transformation of ROI features from 1632 → hidden_dim
2. **3D Brain Map Construction**: Maps ROI features to voxel space using cluster indices
3. **Spatial Block Tokenization**: Divides 3D volume into overlapping blocks and creates tokens
4. **Transformer Encoder**: Processes spatial tokens with multi-head attention
5. **Global Readout**: Average pools all tokens to create a single feature vector
6. **Classification**: Linear classifier for binary classification

### Configuration
```python
hidden_dim = 16           # Projection dimension
num_heads = 4             # Attention heads
num_layers = 1            # Transformer blocks
block_size = 9            # 3D block size (9×9×9)
block_stride = 5          # Block stride
dropout = 0.5             # Dropout rate
```

## Results Interpretation

### Training Output
- **train_log.txt** - Epoch-by-epoch metrics showing:
  - Training loss and accuracy
  - Validation loss and accuracy
  - Learning rate
  - Model checkpoint saves

### Evaluation Output
- **eval_log.txt** - Complete evaluation metrics:
  - Test loss and accuracy
  - Classification report (precision, recall, F1-score)
  - Confusion matrix with TP, TN, FP, FN
  - Per-class accuracy

- **confusion_matrix.jpg** - Visual heatmap of confusion matrix with values and TN/FP/FN/TP legend

- **evaluation_metrics.jpg** - Bar charts showing:
  - Test loss
  - Test accuracy and per-class accuracies

## Advanced: Cross-Validation

For 5-fold cross-validation with repeated runs, use the notebook implementation in `brain_net.ipynb`:

```python
# 5-Fold CV with 3 runs per fold
NUM_FOLDS = 5
NUM_RUNS_PER_FOLD = 3

# Results aggregation by majority voting
# Ties broken by lowest training loss
```

## Troubleshooting

### Common Issues

**1. "No label file found" error in data_prep.py**
- Ensure `label.mat` is in the `toy_data/` directory
- Check file naming convention

**2. Shape mismatch errors**
- Verify feature matrices are (400, 1632)
- Check cluster index matrices are (45, 54, 45)
- Run data_prep.py to identify and remove problematic files

**3. CUDA out of memory**
- Reduce `--batch_size` (try 8 or 4)
- Reduce `--num_epochs` for testing
- Use `--device cpu` for CPU training

**4. Poor model performance**
- Increase `--num_epochs` (default 50 may be too short)
- Decrease `--learning_rate` for stability
- Increase `--weight_decay` for regularization
- Check data splits are stratified properly

## File Descriptions

| File | Purpose |
|------|---------|
| `train.py` | Training script with CLI arguments |
| `evaluate.py` | Evaluation script with visualization |
| `brainnet.py` | Model architecture definitions |
| `dataset.py` | Dataset loading utilities |
| `data_prep.py` | Data preparation and validation |
| `brain_net.ipynb` | Jupyter notebook with full pipeline + CV |
| `requirements.txt` | Python dependencies |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{brainnet2024,
  title={Atlas-free Brain Network Transformer},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `training_runs/` or `evaluation_runs/`
3. Check `train_log.txt` and `eval_log.txt` for detailed error messages

## Quick Start Summary

```bash
# 1. Clone and setup
git clone https://github.com/shourovj/brainnet-transformer.git
cd brainnet-transformer
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data
mkdir toy_data
cp /path/to/your/*.mat toy_data/
python data_prep.py

# 3. Train model
python train.py --num_epochs 50 --batch_size 16

# 4. Evaluate
python evaluate.py --checkpoint_path ./training_runs/run_<timestamp>/best_model.pth

# 5. View results
# Check training_runs/run_<timestamp>/training_curves.jpg
# Check evaluation_runs/run_<timestamp>/ for evaluation metrics
```

---

**Last Updated:** January 2025  
**Version:** 1.0
