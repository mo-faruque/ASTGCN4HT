import os
import os
import sys # Added for stdout redirection
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time # Added for timing
import warnings
import optuna # Added for hyperparameter tuning
from torch_geometric.data import Data # DataLoader moved
from torch_geometric.loader import DataLoader # Correct import for DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool  # Added proper pooling function
from pyverilog.vparser.parser import parse
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
from collections import defaultdict
import random  # Added for setting random seed

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED) # For GPU reproducibility

# Suppress specific pyverilog warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyverilog")

class ASTGenerator:
    """New AST graph generator class"""
    def __init__(self):
        self.type_dict = {}
        self.next_type_id = 0

    def _get_type_id(self, node_type):
        """Get or create type ID for a node type"""
        node_type = node_type.lower()
        
        # Handle common variations
        if node_type.endswith('statement'):
            node_type = node_type[:-9]  # Remove 'statement'
        elif node_type.endswith('substitution'):
            node_type = node_type[:-12]  # Remove 'substitution'
        elif node_type == 'module':
            node_type = 'moduledef'
            
        if node_type not in self.type_dict:
            self.type_dict[node_type] = self.next_type_id
            self.next_type_id += 1
            
        return self.type_dict[node_type]

    def parse_to_ast_graph(self, filepath):
        """Parse Verilog file into AST graph"""
        try:
            # Skip files with known include dependencies
            if 'wb_conmax' in filepath:
                print("Skipping (known include dependencies)")
                return None
                
            # Suppress pyverilog parser warnings
            import contextlib
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                ast, _ = parse([filepath], debug=False)
            
            G = nx.DiGraph()
            self._build_ast_graph(G, ast, parent=None)
            
            # Print type dictionary after first file is parsed
            if not hasattr(self, 'dict_printed'):
                print("\nGenerated AST Type Dictionary:")
                for typ, idx in sorted(self.type_dict.items(), key=lambda x: x[1]):
                    print(f"{idx}: {typ}")
                self.dict_printed = True
                
            return G
        except Exception as e:
            print(f"Skipping (parse error: {str(e)})")
            return None

    def visualize_sample_graph(self, G, filename):
        """Save a sample graph visualization"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, 
               labels={n: str(G.nodes[n]['type']) for n in G.nodes()},
               node_size=500, font_size=8)
        plt.savefig(filename)
        plt.close()

    def export_graph_to_json(self, G, json_filepath):
        """Exports the graph data to a JSON file."""
        import json
        from networkx.readwrite import json_graph
        data = json_graph.node_link_data(G)
        with open(json_filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"AST graph data saved to {json_filepath}")

    def _build_ast_graph(self, G, node, parent, depth=0):
        """Recursively build AST graph with structural features"""
        node_type = node.__class__.__name__
        node_id = str(id(node))
        
        # Get or create type ID
        type_id = self._get_type_id(node_type)
        
        # Add node with enhanced features - now using children_types properly
        children_types = []
        for child in node.children():
            child_type = child.__class__.__name__.lower()
            child_type_id = self._get_type_id(child_type)
            children_types.append(child_type_id)
        
        # Store average of children type IDs as a feature if there are children
        avg_child_type = sum(children_types) / len(children_types) if children_types else 0
        
        G.add_node(node_id, 
                 type=type_id,
                 avg_child_type=avg_child_type,
                 depth=depth,
                 num_children=len(children_types),
                 node_type=node_type.lower())
        
        if parent:
            G.add_edge(parent, node_id)
            
        # Recursively process children
        for child in node.children():
            self._build_ast_graph(G, child, node_id, depth+1)

class TrojanGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, arch='gcn', dropout=0.5):
        super().__init__()
        self.arch = arch
        self.dropout = nn.Dropout(dropout)
        
        if arch == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif arch == 'gat':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
        elif arch == 'graphsage':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif arch == 'gin':
            self.conv1 = GINConv(nn.Linear(in_channels, hidden_channels), train_eps=True)
            self.conv2 = GINConv(nn.Linear(hidden_channels, hidden_channels), train_eps=True)
        
        # MLP classifier parameters are now passed directly
        # classifier_hidden_dim, classifier_layers, classifier_dropout_rate
        # These will be passed from the Optuna objective function or args

        # Ensure classifier_layers is at least 1
        actual_classifier_layers = max(1, self.classifier_layers_arg)

        mlp_layers_list = []
        current_dim = hidden_channels
        if actual_classifier_layers == 1:
            mlp_layers_list.append(nn.Linear(current_dim, 1))
        else:
            for i in range(actual_classifier_layers - 1):
                mlp_layers_list.append(nn.Linear(current_dim, self.classifier_hidden_dim_arg))
                mlp_layers_list.append(nn.ReLU())
                mlp_layers_list.append(nn.Dropout(self.classifier_dropout_rate_arg))
                current_dim = self.classifier_hidden_dim_arg
            mlp_layers_list.append(nn.Linear(current_dim, 1)) # Output layer
            
        self.classifier = nn.Sequential(*mlp_layers_list)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        
        # Fixed: Use proper global pooling
        # If batch is None (for a single graph), create a batch of all zeros
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
        return torch.sigmoid(self.classifier(x))

def parse_verilog_to_graph(filepath, graph_type='structural'):
    """Modified to support both graph types"""
    if graph_type == 'ast':
        generator = ASTGenerator()
        return generator.parse_to_ast_graph(filepath)
    else:  # structural
        try:
            # Original structural graph parsing logic
            ast, _ = parse([filepath], preprocess_include=['.'], preprocess_define=[])
            G = nx.DiGraph()
            
            for module in ast.description.definitions:
                if module.__class__.__name__ == 'ModuleDef':
                    module_name = module.name
                    G.add_node(module_name, type='module')
                    
                    for item in module.items:
                        if item.__class__.__name__ == 'Decl':
                            for var in item.list:
                                if hasattr(var, 'name'):
                                    G.add_node(var.name, type='wire')
                                    G.add_edge(module_name, var.name)
                        elif item.__class__.__name__ == 'InstanceList':
                            for inst in item.instances:
                                if hasattr(inst, 'name'):
                                    gate_type = inst.module.lower().replace('_','')
                                    G.add_node(inst.name, type='instance', gate_type=gate_type)
                                    G.add_edge(module_name, inst.name)
                                    for port in inst.portlist:
                                        if hasattr(port.argname, 'name'):
                                            G.add_edge(inst.name, port.argname.name)
            return G
        except Exception as e:
            print(f"Error parsing {filepath}: {str(e)}")
            return None

def graph_to_data(G, label, graph_type='structural'):
    """Modified to handle both graph types properly"""
    if graph_type == 'ast':
        # AST node features: Fixed to include multiple features
        node_features = []
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        for node in G.nodes():
            # Create a more meaningful feature vector for AST nodes
            type_id = G.nodes[node]['type']
            depth = G.nodes[node]['depth']
            num_children = G.nodes[node]['num_children']
            avg_child_type = G.nodes[node]['avg_child_type']
            
            # One-hot encode the type (expand as needed based on dataset)
            max_types = 50  # Assuming we don't have more than 50 types
            type_onehot = [0] * max_types
            if type_id < max_types:
                type_onehot[type_id] = 1
                
            # Combine all features
            features = [
                *type_onehot,  # Type one-hot encoding
                depth,         # Depth in tree
                num_children,  # Number of children
                avg_child_type # Average child type ID
            ]
            node_features.append(features)
    else:
        # Original structural graph features
        node_features = []
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'wire')
            gate_type = G.nodes[node].get('gate_type', 'unknown')
            in_degree = G.in_degree(node)  # Fixed: Calculate degree directly
            out_degree = G.out_degree(node)  # Fixed: Calculate degree directly

            gate_types = {
                'and':0, 'nand':1, 'or':2, 'nor':3, 'xor':4, 'xnor':5,
                'buf':6, 'not':7, 'dff':8, 'dlatch':9, 'mux':10, 'add':11
            }
            
            gate_type_encoding = [0] * len(gate_types)
            if node_type == 'instance' and gate_type in gate_types:
                gate_type_encoding[gate_types[gate_type]] = 1

            features = [
                1 if node_type == 'module' else 0,
                1 if node_type == 'instance' else 0,
                1 if node_type == 'wire' else 0,
                in_degree,
                out_degree,
                *gate_type_encoding
            ]
            node_features.append(features)
    
    edge_index = []
    for edge in G.edges():
        src, dst = edge
        edge_index.append([node_mapping[src], node_mapping[dst]])
    
    # Create batch tensor for proper pooling
    batch = torch.zeros(len(node_features), dtype=torch.long)
    
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        y=torch.tensor([label], dtype=torch.float),
        batch=batch  # Added batch tensor for proper pooling
    )

def oversample_data(data_list, labels):
    """Perform random oversampling of minority class - fixed implementation"""
    class_counts = {0: 0, 1: 0}
    for label in labels:
        class_counts[label] += 1
    
    if class_counts[0] == class_counts[1]:
        return data_list.copy(), labels.copy()
    
    minority_class = 1 if class_counts[1] < class_counts[0] else 0
    minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
    majority_indices = [i for i, label in enumerate(labels) if label != minority_class]
    
    num_to_add = len(majority_indices) - len(minority_indices)
    indices_to_add = np.random.choice(minority_indices, size=num_to_add, replace=True)
    
    # Create new lists directly with all data
    new_data = data_list.copy()
    new_labels = labels.copy()
    
    # Add oversampled instances
    for i in indices_to_add:
        # Create deep copy of the object to avoid reference issues
        copied_data = Data(
            x=data_list[i].x.clone(),
            edge_index=data_list[i].edge_index.clone(),
            y=data_list[i].y.clone(),
            batch=data_list[i].batch.clone() if hasattr(data_list[i], 'batch') else None
        )
        new_data.append(copied_data)
        new_labels.append(labels[i])
    
    return new_data, new_labels

def main():
    parser = argparse.ArgumentParser(description="GNN for Trojan Detection")
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', default='outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--folds', type=int, default=3, help='Number of cross-validation folds')
    parser.add_argument('--model_type', choices=['gcn', 'gat', 'graphsage', 'gin'], 
                      default='gcn', help='GNN model type')
    parser.add_argument('--graph_type', choices=['structural', 'ast'], 
                      default='structural', help='Type of graph to generate')
    # New arguments for hyperparameter tuning
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for GNN layers')
    parser.add_argument('--classifier_hidden_dim', type=int, default=32, help='Hidden dimension for MLP classifier (used if not running Optuna, or as default range for Optuna)')
    parser.add_argument('--classifier_layers', type=int, default=2, help='Number of layers in MLP classifier (min 1, used if not running Optuna, or as default range for Optuna)')
    parser.add_argument('--classifier_dropout_rate', type=float, default=0.3, help='Dropout rate for MLP classifier (used if not running Optuna, or as default range for Optuna)')
    parser.add_argument('--n_optuna_trials', type=int, default=0, help='Number of Optuna trials to run. If 0, runs a single trial with specified CLI args.')
    args = parser.parse_args()

    # Pass classifier args to TrojanGNN directly now, Optuna will handle suggesting them in objective
    # The direct assignment to TrojanGNN class attributes is removed.
    # Instead, these will be passed to its constructor.

    print(f"\nScript settings:")
    print(f"  Dataset directory: {args.dataset_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Folds: {args.folds}")
    print(f"  Model type: {args.model_type}")
    print(f"  Graph type: {args.graph_type}")
    print(f"  Learning rate: {args.lr}")
    print(f"  GNN Dropout rate: {args.dropout_rate}")
    print(f"  Classifier Hidden Dim: {args.classifier_hidden_dim if args.n_optuna_trials == 0 else '(Optuna controlled)'}")
    print(f"  Classifier Layers: {args.classifier_layers if args.n_optuna_trials == 0 else '(Optuna controlled)'}")
    print(f"  Classifier Dropout: {args.classifier_dropout_rate if args.n_optuna_trials == 0 else '(Optuna controlled)'}")
    print(f"  Optuna Trials: {args.n_optuna_trials}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    
    overall_script_start_time = time.time() 

    # Load and process dataset (do this once)
    print("Loading and processing dataset...")
    data_load_start_time = time.time()
    data_list_global = [] # Use a different name to avoid conflict if main is called multiple times
    for trojan_dir in ['TjIn', 'TjFree']:
        label = 1 if trojan_dir == 'TjIn' else 0
        dir_path = os.path.join(args.dataset_dir, 'TJ-RTL-toy', trojan_dir)
        
        print(f"Processing {trojan_dir} samples...")
        for benchmark in os.listdir(dir_path):
            verilog_file = os.path.join(dir_path, benchmark, 'topModule.v')
            if os.path.exists(verilog_file):
                print(f"  Parsing {benchmark}...", end=' ')
                G = parse_verilog_to_graph(verilog_file, args.graph_type)
                if G:
                    data_item = graph_to_data(G, label, args.graph_type) # Renamed variable
                    data_list_global.append(data_item)
                    print("Done")
                else:
                    print("Skipped (parse error)")
    
    data_load_time = time.time() - data_load_start_time
    print(f"Dataset loading and processing took {data_load_time:.2f} seconds.")
    if not data_list_global:
        print("No data loaded, exiting.")
        return
    print(f"Time per graph for loading/processing: {data_load_time/len(data_list_global):.4f} seconds.")

    labels_global = [data.y.item() for data in data_list_global] # Use a different name

    if args.n_optuna_trials > 0:
        # Wrapper function to pass args and data to the objective function
        # Optuna's objective function only takes `trial` as an argument.
        def objective_wrapper(trial):
            # TrojanGNN class attributes need to be set for each trial based on Optuna's suggestions
            # This is still a bit of a workaround. Ideally, these are direct constructor args.
            TrojanGNN.classifier_hidden_dim_arg = trial.suggest_int("classifier_hidden_dim", 16, 128)
            TrojanGNN.classifier_layers_arg = trial.suggest_int("classifier_layers", 1, 4) # e.g. 1 to 4 layers
            TrojanGNN.classifier_dropout_rate_arg = trial.suggest_float("classifier_dropout_rate", 0.0, 0.7)
            
            # Also suggest GNN dropout and learning rate
            gnn_dropout = trial.suggest_float("gnn_dropout", 0.1, 0.7)
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            
            # Create a temporary args-like object or dict for this trial's HPs
            trial_args = argparse.Namespace(**vars(args)) # Copy fixed args
            trial_args.dropout_rate = gnn_dropout
            trial_args.lr = lr
            # The classifier HPs are set via TrojanGNN class attributes above for now

            return run_training_for_trial(trial, trial_args, data_list_global, labels_global)

        study = optuna.create_study(direction="maximize") # We want to maximize F1 score or accuracy
        study.optimize(objective_wrapper, n_trials=args.n_optuna_trials)

        print("\nOptuna Hyperparameter Tuning Complete!")
        print("Best trial:")
        best_trial_summary = study.best_trial
        print(f"  Value (Maximized Metric): {best_trial_summary.value}")
        print("  Best Hyperparameters: ")
        for key, value in best_trial_summary.params.items():
            print(f"    {key}: {value}")
    else:
        # Run a single trial with CLI arguments if n_optuna_trials is 0
        print("\nRunning a single trial with specified command-line arguments...")
        # Set TrojanGNN class attributes from CLI args for the single run
        TrojanGNN.classifier_hidden_dim_arg = args.classifier_hidden_dim
        TrojanGNN.classifier_layers_arg = args.classifier_layers
        TrojanGNN.classifier_dropout_rate_arg = args.classifier_dropout_rate
        run_training_for_trial(None, args, data_list_global, labels_global) # Pass None for trial

    total_script_time = time.time() - overall_script_start_time
    print(f"\nTotal script execution time: {total_script_time:.2f} seconds.")

def run_training_for_trial(trial, current_args, data_list, labels):
    """
    Encapsulates the training and evaluation logic for a single trial (either Optuna or manual).
    `trial` is an Optuna trial object, or None for a manual run.
    `current_args` contains all necessary arguments (from CLI or Optuna trial).
    `data_list` and `labels` are the pre-loaded dataset.
    Returns the metric to be optimized by Optuna (e.g., mean F1 score).
    """
    print(f"\n--- Starting Trial (Optuna trial: {trial.number if trial else 'Manual'}) ---")
    if trial:
        print(f"Parameters for this trial: {trial.params}")
    else: # Manual run, print relevant HPs from current_args
        print(f"Parameters for this manual run:")
        print(f"  lr: {current_args.lr}")
        print(f"  gnn_dropout: {current_args.dropout_rate}")
        print(f"  classifier_hidden_dim: {TrojanGNN.classifier_hidden_dim_arg}") # Using the class attr
        print(f"  classifier_layers: {TrojanGNN.classifier_layers_arg}")
        print(f"  classifier_dropout_rate: {TrojanGNN.classifier_dropout_rate_arg}")


    # Calculate pos_weight for BCELoss - this depends on the full dataset, so can be outside trial
    clean_count = labels.count(0)
    trojan_count = labels.count(1)
    pos_weight = torch.tensor([clean_count / max(1, trojan_count)], dtype=torch.float)
    
    all_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'confusion_matrices': [], 'train_accuracy': []
    }
    fold_training_times = []
    fold_evaluation_times = []
    
    kf = StratifiedKFold(n_splits=current_args.folds, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_list, labels)):
        print(f"\n--- Fold {fold+1}/{current_args.folds} (Trial: {trial.number if trial else 'Manual'}) ---")
        original_train_data = [data_list[i] for i in train_idx]
        original_train_labels = [labels[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        train_data, train_labels = oversample_data(original_train_data, original_train_labels)
        
        # Model Initialization
        in_channels = data_list[0].x.shape[1]
        # Note: TrojanGNN still picks up classifier HPs from its class attributes
        # which are set in objective_wrapper (for Optuna) or main (for single run)
        model = TrojanGNN(in_channels=in_channels, hidden_channels=64, 
                          arch=current_args.model_type, dropout=current_args.dropout_rate)
        optimizer = optim.Adam(model.parameters(), lr=current_args.lr)
        criterion = nn.BCELoss(weight=pos_weight)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        epoch_train_start_time = time.time()
        for epoch in range(current_args.epochs):
            model.train()
            total_loss = 0
            for batch_data in train_loader:
                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output.squeeze(1), batch_data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_data.num_graphs
            if (epoch+1) % 10 == 0 or epoch == current_args.epochs -1 : # Print last epoch too
                 print(f"Epoch {epoch+1}/{current_args.epochs}, Loss: {total_loss/len(train_data):.4f}")
        
        fold_training_time = time.time() - epoch_train_start_time
        fold_training_times.append(fold_training_time)

        # Training Accuracy
        model.eval()
        train_correct = 0
        train_total_samples = 0
        with torch.no_grad():
            for batch_data in train_loader:
                output = model(batch_data)
                pred = (output.squeeze() > 0.5).float()
                if pred.shape != batch_data.y.shape:
                    if output.squeeze().shape == batch_data.y.shape: pred = (output.squeeze() > 0.5).float()
                    else: print(f"Warning: Shape mismatch in train accuracy. Pred: {pred.shape}, Y: {batch_data.y.shape}")
                train_correct += (pred == batch_data.y).sum().item()
                train_total_samples += batch_data.y.size(0)
        train_accuracy_fold = train_correct / train_total_samples if train_total_samples > 0 else 0
        all_metrics['train_accuracy'].append(train_accuracy_fold)

        # Evaluation
        eval_start_time = time.time()
        test_loss = 0
        correct = 0
        preds, truths = [], []
        with torch.no_grad():
            for batch_data_test in test_loader:
                output = model(batch_data_test)
                test_loss += criterion(output.squeeze(1), batch_data_test.y).item() * batch_data_test.num_graphs
                pred = (output.squeeze() > 0.5).float()
                if pred.shape != batch_data_test.y.shape:
                    if output.squeeze().shape == batch_data_test.y.shape: pred = (output.squeeze() > 0.5).float()
                    else: print(f"Warning: Shape mismatch in test accuracy. Pred: {pred.shape}, Y: {batch_data_test.y.shape}")
                correct += (pred == batch_data_test.y).sum().item()
                preds.extend(pred.cpu().numpy().flatten().tolist())
                truths.extend(batch_data_test.y.cpu().numpy().flatten().tolist())
        
        fold_evaluation_time = time.time() - eval_start_time
        fold_evaluation_times.append(fold_evaluation_time)
        test_loss /= len(test_data)

        accuracy = accuracy_score(truths, preds)
        precision = precision_score(truths, preds, zero_division=0)
        recall = recall_score(truths, preds, zero_division=0)
        f1 = f1_score(truths, preds, zero_division=0)
        
        all_metrics['accuracy'].append(accuracy)
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['f1'].append(f1)
        all_metrics['confusion_matrices'].append(confusion_matrix(truths, preds))

        print(f'Fold Metrics: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Train Acc: {train_accuracy_fold:.4f}')
        # ... (print confusion matrix, timing per fold)
        print(f"Fold Training Time: {fold_training_time:.2f}s ({fold_training_time/len(train_data) if len(train_data) > 0 else 0:.4f}s/sample)")
        print(f"Fold Evaluation Time: {fold_evaluation_time:.2f}s ({fold_evaluation_time/len(test_data) if len(test_data) > 0 else 0:.4f}s/sample)")


    # Averaged metrics for this trial
    mean_f1 = np.mean(all_metrics["f1"])
    mean_accuracy = np.mean(all_metrics["accuracy"])
    print(f"\n--- Trial Summary (Optuna trial: {trial.number if trial else 'Manual'}) ---")
    print(f'Average Test Accuracy: {mean_accuracy:.4f}')
    print(f'Average Train Accuracy: {np.mean(all_metrics["train_accuracy"]):.4f}')
    print(f'Average F1 Score: {mean_f1:.4f}')
    # ... (print other averaged metrics if needed)

    if trial: # For Optuna, report intermediate values if desired (for pruning)
        trial.report(mean_f1, step=current_args.epochs) # Report F1 score at the end of all epochs for this trial
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return mean_f1 # Optuna will maximize this

# Helper class to write to multiple streams (e.g., console and file)
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Flush ensure things appear in console immediately
    def flush(self):
        for f in self.files:
            f.flush()

if __name__ == "__main__":
    original_stdout = sys.stdout
    # Decide where summary.txt goes. Let's try putting it in the output_dir if specified.
    # We need to parse args *before* setting up Tee if we want to use output_dir.
    # Temporarily parse args just for output_dir, or accept summary.txt in CWD.
    # For simplicity now, let's keep summary.txt in the CWD.
    output_file_path = 'summary.txt'
    
    print(f"Script output will also be saved to {os.path.abspath(output_file_path)}") # To console

    try:
        with open(output_file_path, 'w') as f_summary:
            # Create a Tee object to write to both original stdout and the file
            tee = Tee(original_stdout, f_summary) 
            sys.stdout = tee # Redirect stdout to the Tee object
            main() # All prints from main() and its calls will now go to both streams
    finally:
        sys.stdout = original_stdout # Restore original stdout regardless of errors
        
    print(f"\nScript execution finished. Full output saved to {os.path.abspath(output_file_path)}") # To console
