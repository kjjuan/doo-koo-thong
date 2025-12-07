### Folder Overview & Structure

The project follows the standard Kedro structure, organized to separate configuration, data, and source code. 

Contributors:

Matthew Doo Zheng En: oodmatt@gmail.com

Koo Jia Juan: koojiajuan123@gmail.com

Thong Joon Lee: joonleethong@gmail.com



```t
egt309/
├── conf/                       # Configuration files
│   └── base/               
│       ├── catalog.yml         # Registry of all data sources and sinks
│       └── parameters.yml      # Global parameters
├── data/                       # Data storage (Local only, Gitignored)
│   ├── 01_raw                  # Immutable source data (bmarket.csv)
│   ├── 02_intermediate         # -
│   ├── 03_primary              # Cleaned data (missing values filled, types corrected)
│   ├── 04_feature              # Engineered features (One-Hot Encoded, Scaled)
│   ├── 05_model_input          # Train/Test splits and Preprocessed model
│   ├── 06_models               # Serialized models (.pkl)
│   ├── 07_model_output         # -
│   └── 08_reporting            # Confusion matrices and metric logs
├── notebooks/                  # Jupyter notebooks for experiments (eda.ipynb)
|   ├── eda.ipynb
|   └── eda.pdf
├── src/                        # Source code
│   └── egt309/
│       └── pipelines/          # Modular pipeline logic
│           ├── data_prep       # Cleaning and Feature Engineering
│           ├── data_science    # Model training and splitting
│           └── reporting       # Evaluation visualizations
|       └── pipeline_registry.py# Register Project Pipelines
├── run.sh                      # Helper script for easy execution
├── .gitignore                  # Files to ignore when pushing updates to github
└── requirements.txt            # Project dependencies
```

### Prerequisites
1. Python 3.8+
2. Pip
3. Docker 

### Installation

Ensure you are in the project root and have dependencies installed, input ```pip install -r requirements.txt``` in the terminal. 

To run the whole Kedro pipeline, run ```./run.sh```, or to specify a pipeline to run, e.g. the 'data_science' pipeline, run ```./run.sh data_science```

### Hyperparameters
To adjust hyperparameters (e.g., model learning rates, train/test split ratios, or file paths),
1. Navigate to ```conf/base/.```
2. Edit parameters_data_science.yml to change model settings (e.g., XGBoost depth, Random Forest estimators).
3. Edit parameters.yml or catalog.yml to change file paths or reporting directories.

### Kedro Pipeline Logical Flow

This pipeline is organized into three main stages. The flow is described below using a flow chart to show dependency and data movement.

<img src="https://github.com/kjjuan/doo-koo-thong/blob/main/Logical_Flow.png" alt="Pipeline Flow Chart">

### EDA Findings and choice of Models
1. Target Imbalance

The dataset is highly imbalanced. Only 11.3% of clients subscribed to the term deposit, while 88.7% did not.

Models are evaluated using Confusion Matrices and F1-scores rather than just accuracy to ensure the minority class is detected.

2. Demographics

For Age, subscription rates are highest among the very young (16-25) and the elderly (66+). The middle-aged workforce has lower subscription rates. 

Hence for Feature Engineering, We engineered an Age Group feature to capture the non-linear relationship between age and subscription likelihood.

For Jobs, Students and Retired individuals are the most likely to subscribe.

For Education, Higher education levels correlate with higher subscription rates.

### Feature Processing Summary

The following table summarizes how specific features are processed within the pipeline to address data quality issues identified during the analysis phase.

| Feature Name | Issue Identified | Transformation Applied in Pipeline |
| ---------- | ---------- | ---------- |
| **Age** | Contains string "years" and unrealistic outliers (e.g., 150). | 1. Stripped string "years". <br> 2. Removed rows where Age = 150. <br> 3. Binned values into `Age Group` (16-25, 26-35, etc.). |
| **Contact Method** | Inconsistent naming ("Cell" vs "cellular", "Telephone" vs "telephone"). | Standardized values to lowercase ('cellular', 'telephone') to merge categories. |
| **Campaign Calls** | Contained negative values (e.g., -1) due to formatting errors. | Converted to absolute integers to ensure valid counts. |
| **Previous Contact Days** | Represented as `999` if the client was never contacted previously. | 1. Created binary flag `WasContactedBefore`. <br> 2. Replaced `999` with `0` in a new feature `PreviouslyContacted`. |
| **Housing & Personal Loan** | Contained missing/null values. | Filled missing values with the category `"unknown"` to preserve data integrity. |
| **Categorical Columns** | Models require numeric input. | Applied **One-Hot Encoding** to Occupation, Marital Status, and Education features. |


