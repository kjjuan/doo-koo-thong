"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node

from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_and_process,              # runs raw data cleaning
                inputs="df_raw",                     # input dataset name in catalog
                outputs="df_cleaned",                # cleaned dataset name
                name="clean_and_process_data", # kedro run --node "clean_and_process_data"
                tags=["data_cleaning"],  # kedro run --tags "data_cleaning"
            ),
            node(
                func=engineer_and_prepare,           # runs feature engineering
                inputs="df_cleaned",                 # takes cleaned data
                outputs="df_prep",                   # outputs prepared data
                name="engineer_features",
                tags=["feature_engineering"],
            ),
            node(
                func=split_data,                     # splits into train/test
                inputs=["df_prep", "params:data_split_options"], # uses prepared data + split params
                outputs=["X_train", "X_test", "y_train", "y_test"], # train/test features and targets
                name="split_data_node",
                tags=["data_splitting"],
            ),
            node(
                func=preprocess_data,                # applies OHE + ColumnTransformer
                inputs=["X_train", "X_test", "params:features"],   # raw train/test + feature config
                outputs=[
                    "X_train_processed",             # encoded train features
                    "X_test_processed",              # encoded test features
                    "preprocessor",                  # fitted transformer for reuse
                    "processed_features"             # list of feature names after encoding
                ],
                name="preprocessing_data",
                tags=["data_preprocessing"],
            )
        ],
    )
