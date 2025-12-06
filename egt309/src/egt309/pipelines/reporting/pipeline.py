"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, pipeline  
from .nodes import evaluate_models, plot_confusion_matrices 

def create_pipeline(**kwargs) -> pipeline:
    return pipeline(
        [
            # Node for model evaluation
            node(
                func=evaluate_models,
                inputs=["trained_models", "X_test", "y_test", "params:thresholds"],
                outputs="evaluation_results",
                name="evaluate_models_node",
            ),
            # Evaluation of the models (Confusion Matrix)
            node(
                func=plot_confusion_matrices,
                inputs=["evaluation_results"], # Takes the DF from the previous node
                outputs=None, # Returns nothing, just saves files as side effect
                name="plot_confusion_matrices_node",
            ),
        ]
    )
