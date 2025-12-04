"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, pipeline  
from .nodes import *

def create_pipeline(**kwargs) -> pipeline:
    return pipeline(
        [
            node(
                func=plot_confusion_matrices,
                inputs=["evaluation_results"], # Takes the DF from the previous node
                outputs=None, # Returns nothing, just saves files as side effect
                name="plot_confusion_matrices_node",
            ),
        ]
    )
