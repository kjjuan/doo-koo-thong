"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node

from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the data cleaning and preparation pipeline.
    """
    return Pipeline(
        [
            node(
                func=clean_and_process,
                inputs="df_raw",
                outputs="df_cleaned",
                name="clean_and_process_data", # kedro run --node "clean_and_process_data"
                tags=["data_cleaning"],  # kedro run --tags "data_cleaning"
            ),
            node(
                func=engineer_and_prepare,
                inputs="df_cleaned",
                outputs="df_prep",
                name="engineer_features",
                tags=["feature_engineering"],
            ),
            node(
                func=split_data,
                inputs=["df_prep", "params:data_split_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
                tags=["data_splitting"],
            ),
            node(
                func=preprocess_data,
                inputs=["X_train", "X_test", "params:features"],
                outputs=[
                    "X_train_processed",
                    "X_test_processed",
                    "preprocessor",
                    "processed_features"
                ],
                name="preprocessing_data",
                tags=["data_preprocessing"],
            )
        ],
    )