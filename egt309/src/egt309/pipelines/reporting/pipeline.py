
"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node

from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=plot_confusion_matrices,
                inputs=["model_evaluation_metrics"], # Takes the DF from the previous node
                outputs=None, # Returns nothing, just saves files as side effect
                name="plot_confusion_matrices_node",
            ),

        ],
    )
