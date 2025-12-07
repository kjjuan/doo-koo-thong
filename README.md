### Folder Overview & Structure

The project follows the standard Kedro structure, organized to separate configuration, data, and source code. 

Contributors + Contributions:

Matthew Doo Zheng En: oodmatt@gmail.com
- Data Understanding, EDA, Data Analysis, Docker, Bash File

Koo Jia Juan: koojiajuan123@gmail.com
- Data Understanding, Data Cleaning, Data Preparation, Model Fine-Tuning, Kedro

Thong Joon Lee: joonleethong@gmail.com
- Data Understanding, Data Analysis, Data Preparation, ML reasearch, ML Model Selection, Model Training, Data & Results Analysis/Overview, Model Fine-Tuning


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

If you have a windows system, you will need wsl or ubuntu to input these commands, otherwise if you have a macos or linux system, ignore the commands below:

```git clone 

python3 -m venv venv

source venv/bin/activate

cd egt309
 ```

Ensure you are in the project root and have dependencies installed, input ```pip install -r requirements.txt``` in the terminal. 

To run the whole Kedro pipeline, run ```bash run.sh```, or to specify a pipeline to run, e.g. the 'data_science' pipeline, run ```bash run.sh data_science```

### Hyperparameters
To adjust hyperparameters (e.g., model learning rates, train/test split ratios, or file paths),
1. Navigate to ```conf/base/```
2. Edit parameters.yml to change model settings (e.g., XGBoost depth, Random Forest estimators).
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

### Choice of Models

We trained four different models to understand which type works best for predicting whether a client will subscribe. Each model was chosen for a specific reason based on what we saw from the EDA.


1. Logistic Regression

This was used as our baseline model. It is simple, easy to interpret, and good for showing basic trends. However, our data had many non-linear patterns, so this model could not capture all relationships. It helped us see how much improvement the more complex models could achieve.

2. Random Forest

Random Forest was chosen because it handles non-linear patterns well and works smoothly with one-hot encoded categorical features. It is also more stable and less likely to overfit than a single decision tree. It usually gives strong baseline performance, so we included it to compare against the boosted models.

3. Gradient Boosting

We included Gradient Boosting to capture more complex interactions than Random Forest. Because it builds trees one at a time and learns from previous mistakes, it can find patterns that simpler models miss. It generally performs well on structured datasets like ours and helped us understand the gains from boosted models.

4. XGBoost

This model was added because it is one of the best-performing boosting algorithms for tabular data. It has built-in ways to handle imbalanced data and provides many parameters to fine-tune performance. After testing, XGBoost gave the strongest overall results, especially when tuned properly.


### Model Evaluation and Tuning Process
Because the dataset is highly imbalanced (only 11.3% subscribed), accuracy alone was not useful. Instead, we evaluated models using:

- F1-score (balances precision and recall)

- Precision (how many predicted “yes” were correct)

- Recall (how many actual “yes” we detected)

- Confusion matrices (to see where the model is making mistakes)

These metrics helped us compare models more fairly.

We performed many rounds of experimentation to improve the results. This included:


1. Threshold Testing

We changed the prediction threshold across many values, from 0.1 up to 0.9. This helped us see how the model behaves when we try to increase recall or precision. Because the default 0.5 threshold is not always suitable for imbalanced data, this step was important.


3. SMOTE vs. No SMOTE

We trained the models multiple times with and without SMOTE.

- SMOTE helped increase recall for some models but sometimes reduced precision.

- Tree models like Random Forest did not always benefit from oversampling.

- XGBoost performed better using scale_pos_weight instead of SMOTE.

This showed us that imbalance handling is not one-size-fits-all and needs testing.


3. Hyperparameter Tuning

We adjusted many parameters over many trials to improve performance. Examples include:

- XGBoost: max_depth, n_estimators, learning rate, scale_pos_weight, min_child_weight

- Random Forest: n_estimators, max_depth, min_samples_leaf, class_weight

- Logistic Regression: C value, penalty type, class_weight

We changed these parameters repeatedly until improvements became small. This tuning process took many iterations and helped us push the models to produce better F1-scores.


4. Final Comparison

After tuning all models, XGBoost had the best combination of recall, precision, and F1-score.

Random Forest performed well but did not surpass XGBoost.

Gradient Boosting improved over the baseline but still lower than XGBoost.

Logistic Regression remained the weakest but was important as our baseline.


### References and Citation

OpenAI. (2025). ChatGPT (version 5.1) [Large language model]. https://chat.openai.com/

Google. (2025). Gemini (version 2.0) [Large language model]. https://gemini.google.com/

