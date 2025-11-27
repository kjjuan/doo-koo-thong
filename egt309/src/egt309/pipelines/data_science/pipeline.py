"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                input=["subscribe", "params:insert here"],
                output=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node"
            )
            node(
                func=train_model,
                input=["X_train", "X_test", "y_train", "y_test"],
                output=
            )
    ])
