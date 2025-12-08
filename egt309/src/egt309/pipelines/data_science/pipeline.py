"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node

from .nodes import *


# done by matthew
def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_models,                 # node that trains all ML models
                inputs=[
                    "X_train",                     # training features (not yet encoded)
                    "y_train",                     # training labels
                    "preprocessor",                # fitted ColumnTransformer from data_prep
                    "parameters"                   # full parameter dictionary (hyperparams + split options)
                ],
                outputs={
                    "LogReg": "trained_logreg_model",           # LogReg key saves to this catalog entry
                    "RandomForest": "trained_rf_model",         # RandomForest key saves to this catalog entry
                    "GradientBoosting": "trained_gb_model",
                    "XGBoost": "trained_xgb_model",
                },          # dictionary of trained model pipelines
                name="train_models_node",
                tags=["model_training"],           # can run this node by tag
            ),
            node(
                func=evaluate_models,              # node that evaluates all models
                inputs=[
                    "trained_logreg_model",      
                    "trained_rf_model",         
                    "trained_gb_model",          
                    "trained_xgb_model",            
                    "X_test",                      # raw test features
                    "y_test",                      # test labels
                    "params:evaluation_thresholds" # list of thresholds from parameters.yml
                ],
                outputs="model_evaluation_metrics",# DataFrame of metrics for all models/thresholds
                name="evaluate_models_node",
                tags=["model_training"],
            ),
        ],
    )
