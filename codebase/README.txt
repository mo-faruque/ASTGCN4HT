Hardware Trojan Detection using Graph Neural Networks

## Overview
This project implements a Graph Neural Network (GNN) approach for detecting hardware Trojans in Verilog designs. The main.py script converts hardware designs into graph representations and uses GNN models to classify them as either Trojan-free or Trojan-infected.

## Features
- Converts Verilog designs into two types of graph representations:
  - Abstract Syntax Tree (AST) graphs
- Supports multiple GNN architectures:
  - Graph Convolutional Networks (GCN)
  - Graph Attention Networks (GAT)
  - GraphSAGE
  - Graph Isomorphism Networks (GIN)
- Implements cross-validation for robust evaluation
- Includes hyperparameter tuning with Optuna
- Provides detailed metrics (accuracy, precision, recall, F1 score)
- Handles class imbalance through oversampling

## Requirements
- Python 3.6 or higher
- PyTorch
- PyTorch Geometric
- Pyverilog
- NetworkX
- Scikit-learn
- Matplotlib
- Optuna
- NumPy

You can install the required packages using:
```
pip install -r requirements.txt
```

## Usage
```
python main.py --dataset_dir <path_to_dataset> [options]
```

### Required Arguments
- `--dataset_dir`: Path to the dataset directory containing Verilog files

### Optional Arguments
- `--output_dir`: Output directory for results (default: 'outputs')
- `--epochs`: Number of training epochs (default: 50)
- `--folds`: Number of cross-validation folds (default: 3)
- `--model_type`: GNN model type (choices: 'gcn', 'gat', 'graphsage', 'gin', default: 'gcn')
- `--graph_type`: Type of graph to generate (choices: 'structural', 'ast', default: 'structural')
- `--lr`: Learning rate (default: 0.001)
- `--dropout_rate`: Dropout rate for GNN layers (default: 0.5)
- `--classifier_hidden_dim`: Hidden dimension for MLP classifier (default: 32)
- `--classifier_layers`: Number of layers in MLP classifier (default: 2)
- `--classifier_dropout_rate`: Dropout rate for MLP classifier (default: 0.3)
- `--n_optuna_trials`: Number of Optuna trials to run for hyperparameter tuning (default: 0, which means no tuning)

## Example Usage
```
# Run with default parameters
python main.py --dataset_dir combined_dataset/TJ-RTL-toy

# Run with AST graph representation and GAT model
python main.py --dataset_dir combined_dataset/TJ-RTL-toy --graph_type ast --model_type gat

# Run with hyperparameter tuning (10 Optuna trials)
python main.py --dataset_dir combined_dataset/TJ-RTL-toy --n_optuna_trials 10
```

## Output
The script outputs:
1. Training progress and loss for each epoch
2. Evaluation metrics for each fold (accuracy, precision, recall, F1 score)
3. Confusion matrices
4. Average metrics across all folds
5. Training and evaluation times
6. Best hyperparameters (if using Optuna)

All output is saved to a summary.txt file in the current directory.

## Dataset Structure
The script expects the dataset directory to have the following structure:
```
dataset_dir/
├── TJ-RTL-toy/
│   ├── TjFree/
│   │   ├── benchmark1/
│   │   │   └── topModule.v
│   │   ├── benchmark2/
│   │   │   └── topModule.v
│   │   └── ...
│   └── TjIn/
│       ├── benchmark1/
│       │   └── topModule.v
│       ├── benchmark2/
│       │   └── topModule.v
│       └── ...
```

## Notes
- The script sets random seeds for reproducibility
- Class imbalance is handled through oversampling of the minority class
- For hyperparameter tuning, the script optimizes for F1 score
- The script automatically saves all output to summary.txt
