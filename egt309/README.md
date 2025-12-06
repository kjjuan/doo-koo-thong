Folder Overview & Structure

The project follows the standard Kedro structure, organized to separate configuration, data, and source code.

```t
egt309/
├── conf/                   # Configuration files
│   └── base/               
│       ├── catalog.yml     # Registry of all data sources and sinks
│       ├── parameters.yml  # Global parameters
│       └── parameters_*.yml# Pipeline-specific parameters (data_science, reporting, etc.)
├── data/                   # Data storage (Local only, Gitignored)
│   ├── 01_raw              # Immutable source data (bmarket.csv)
│   ├── 02_intermediate     # Cleaned data (missing values filled, types corrected)
│   ├── 03_primary          # Domain-specific data
│   ├── 04_feature          # Engineered features (One-Hot Encoded, Scaled)
│   ├── 05_model_input      # Train/Test splits
│   ├── 06_models           # Serialized models (.pkl)
│   ├── 07_model_output     # Predictions
│   └── 08_reporting        # Confusion matrices and metric logs
├── notebooks/              # Jupyter notebooks for experiments (eda.ipynb)
├── src/                    # Source code
│   └── egt309/
│       └── pipelines/      # Modular pipeline logic
│           ├── data_prep   # Cleaning and Feature Engineering
│           ├── data_science# Model training and splitting
│           └── reporting   # Evaluation visualizations
├── run.sh                  # Helper script for easy execution
└── requirements.txt        # Project dependencies
```

Prerequisites:
1. Python 3.8+
2. Pip
3. Docker 

Installation:
Ensure you are in the project root and have dependencies installed, input ```pip install -r requirements.txt``` in the terminal. 
To run the whole Kedro pipeline, run ```./run.sh```, or to specify a pipeline to run, e.g. the 'data_science' pipeline, run ```./run.sh data_science```

