# ASTGCN4HT Project

## Project Description
This repository contains the ASTGCN4HT course project (EE 364 [Computer Performance Alanysis I, NMSU, Spring 25]), focused on hardware Trojan detection using Verilog files and Python analysis tools. Detailed results and analysis can be found in the report/ directory. The project includes:

- Multiple Verilog implementations (both clean and Trojan-infected versions)
- Python analysis scripts
- Documentation and reports

## Project Structure
```
ASTGCN4HT/
├── codebase/                # Main code directory
│   ├── combined_dataset/    # Verilog implementations
│   │   ├── TJ-RTL-toy/      # Trojan-related implementations
│   │   │   ├── TjFree/      # Clean implementations
│   │   │   └── TjIn/        # Trojan-infected implementations
│   ├── main.py              # Main analysis script
│   ├── requirements.txt     # Python dependencies
│   └── ...                  
├── presentation/            # Presentation materials
└── report/                  # Project reports
```

## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/mo-faruque/ASTGCN4HT.git
```

2. Install Python dependencies:
```bash
pip install -r codebase/requirements.txt
```

3. Run the analysis:
```bash
python codebase/main.py
```

## Key Findings (from Optuna Hyperparameter Tuning)

The following are the results from the hyperparameter tuning process for the hardware Trojan detection model:

**Best Trial Summary:**
- **Average Test Accuracy**: 0.8825
- **Average Train Accuracy**: 0.8976
- **Average F1 Score**: 0.9311

**Best Hyperparameters:**
- `classifier_hidden_dim`: 32
- `classifier_layers`: 4
- `classifier_dropout_rate`: 0.217052023813549
- `gnn_dropout`: 0.6211861174086543
- `lr`: 0.003447510318746691

For complete training logs and detailed fold metrics, refer to `codebase/summary.txt` and the full reports in the `report/` directory.

## License
This project is currently unlicensed. Please contact the repository owner for usage permissions.
