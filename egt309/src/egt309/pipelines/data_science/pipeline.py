"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node

from .nodes import *

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
                outputs="trained_models",          # dictionary of trained model pipelines
                name="train_models_node",
                tags=["model_training"],           # can run this node by tag
            ),
            node(
                func=evaluate_models,              # node that evaluates all models
                inputs=[
                    "trained_models",              # dict of fitted models from previous node
                    "X_test",                      # raw test features
                    "y_test",                      # test labels
                    "params:evaluation_thresholds" # list of thresholds from parameters.yml
                ],
                outputs="model_evaluation_metrics",# DataFrame of metrics for all models/thresholds
                name="evaluate_models_node",
                tags=["model_training"],
            ),
            node(
                func=plot_confusion_matrices,
                inputs=["model_evaluation_metrics"], # Takes the DF from the previous node
                outputs=None, # Returns nothing, just saves files as side effect
                name="plot_confusion_matrices_node",
            ),

        ],
    )
