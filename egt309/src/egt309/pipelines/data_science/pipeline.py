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
                func=train_models,
                inputs=[
                    "X_train",
                    "y_train",
                    "preprocessor",
                    "parameters"
                ],
                outputs="trained_models",
                name="train_models_node",
                tags=["model_training"],
            ),
            node(
                func=evaluate_models,
                inputs=[
                    "trained_models",
                    "X_test",
                    "y_test",
                    "params:evaluation_thresholds"
                ],
                outputs="model_evaluation_metrics",
                name="evaluate_models_node",
                tags=["model_evaluation"],
            )
        ],
    )