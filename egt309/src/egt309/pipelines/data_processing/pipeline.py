"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=local_raw_data,
            input=""
            output="df",
        ),
        node(
            func=clean_dataset,
            input="df",
            output="df",
        ),
    ])
