"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, pipeline  # noqa


def create_pipeline(**kwargs) -> pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs= ["subscribe", "params:insert here"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs= "model"
            )
        ]
)
