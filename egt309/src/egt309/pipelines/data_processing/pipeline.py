"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, pipeline  # noqa
from .nodes import load_raw_data, clean_dataset


def create_pipeline(**kwargs) -> pipeline:
    return pipeline([
        node(
            func=load_raw_data,
            inputs="df",
            outputs="dfi",
        ),
        node(
            func=clean_dataset,
            inputs="dfi",
            outputs="dfii",
        ),
    ])
