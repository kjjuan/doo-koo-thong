"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer

def train_models(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer, params: dict) -> dict:
    """
    Trains multiple models using an ImbPipeline that includes SMOTE and the preprocessor.
    
    NOTE: X_train/y_train here refer to the already split, but not yet encoded, data.
    The ImbPipeline handles the encoding within its first step ('preprocess').
    """

    model_hyperparameters = params['model_hyperparameters']  # read all model settings from parameters.yml
    data_split_options = params['data_split_options']        # read split options (for random_state)

    y_train = y_train.iloc[:, 0] # Convert back to Series for SMOTE / sklearn models
    
    models = {
        "LogReg": ImbPipeline(steps=[
            ('preprocess', preprocessor),  # apply ColumnTransformer (OHE etc.) inside pipeline
            ('model', LogisticRegression(
                max_iter=model_hyperparameters['logreg']['max_iter'],          # max optimisation steps
                class_weight=model_hyperparameters['logreg']['class_weight'],  # balance yes/no classes
                random_state=data_split_options['random_state']                # reproducible results
            ))
        ]),
        "RandomForest": ImbPipeline(steps=[
            ('preprocess', preprocessor),
            ('model', RandomForestClassifier(
                n_estimators=model_hyperparameters['random_forest']['n_estimators'],      # number of trees
                max_depth=model_hyperparameters['random_forest']['max_depth'],            # tree depth limit
                min_samples_split=model_hyperparameters['random_forest']['min_samples_split'], # min samples to split
                class_weight=model_hyperparameters['random_forest']['class_weight'],      # balance classes
                random_state=data_split_options['random_state']                           # reproducible forest
            ))
        ]),
        "GradientBoosting": ImbPipeline(steps=[
            ('preprocess', preprocessor),
            ('model', GradientBoostingClassifier(
                learning_rate=model_hyperparameters['gradient_boosting']['learning_rate'], # step size per tree
                n_estimators=model_hyperparameters['gradient_boosting']['n_estimators'],   # number of boosting stages
                max_depth=model_hyperparameters['gradient_boosting']['max_depth'],         # depth of each tree
                random_state=data_split_options['random_state']                            # reproducible boosting
            ))
        ]),

        "XGBoost": ImbPipeline(steps=[
            ('preprocess', preprocessor),
            ('model', XGBClassifier(
                    n_estimators=model_hyperparameters['xgboost']['n_estimators'],        # number of boosted trees
                    learning_rate=model_hyperparameters['xgboost']['learning_rate'],      # learning rate / eta
                    max_depth=model_hyperparameters['xgboost']['max_depth'],              # tree depth
                    subsample=model_hyperparameters['xgboost']['subsample'],              # row subsample per tree
                    colsample_bytree=model_hyperparameters['xgboost']['colsample_bytree'],# feature subsample per tree
                    gamma=model_hyperparameters['xgboost']['gamma'],                      # min loss reduction to split
                    scale_pos_weight=model_hyperparameters['xgboost']['scale_pos_weight'],# upweight positive class
                    reg_lambda=model_hyperparameters['xgboost']['reg_lambda'],            # L2 regularisation strength
                    eval_metric=model_hyperparameters['xgboost']['eval_metric'],          # training metric (logloss)
                    random_state=data_split_options['random_state']                       # reproducible boosting
            ))
        ])

    }
    
    trained_models = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)   # fit each pipeline on the same train data
        trained_models[name] = mdl  # store trained model by name
        
    return trained_models


def evaluate_models(logreg_model: ImbPipeline, rf_model: ImbPipeline, gb_model: ImbPipeline, xgb_model: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series, thresholds: list) -> pd.DataFrame:
    """
    Evaluates trained models across multiple probability thresholds and compiles results.
    """
    trained_models = {
        "LogReg": logreg_model,
        "RandomForest": rf_model,
        "GradientBoosting": gb_model,
        "XGBoost": xgb_model,
    }
    
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    # Convert back to Series
    all_results = []  # will collect metrics for all models and thresholds
    
    for thr in thresholds:
        rows = []  # metrics for this specific threshold
        for name, mdl in trained_models.items():
            # Get probabilities for the positive class (1)
            probs = mdl.predict_proba(X_test)[:, 1]  # use predicted probability of "yes"
            
            # Apply the threshold to get final predictions
            y_pred = (probs >= thr).astype(int)      # convert prob → 0/1 label
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)                              # overall correctness
            prec = precision_score(y_test, y_pred, zero_division=0)           # how many predicted yes are correct
            rec = recall_score(y_test, y_pred, zero_division=0)               # how many actual yes we catch
            f1 = f1_score(y_test, y_pred, zero_division=0)                    # balance of precision & recall
            cm = confusion_matrix(y_test, y_pred)                             # TP/FP/FN/TN counts

            rows.append({
                "Threshold": thr,                  # probability threshold used
                "Model": name,                     # model name (LogReg / RandomForest / etc.)
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "Confusion Matrix": cm.tolist()    # store as list so it serialises to JSON / CSV
            })
        
        all_results.extend(rows)  # add all models for this threshold

    return pd.DataFrame(all_results)  # final table of all evaluation results
