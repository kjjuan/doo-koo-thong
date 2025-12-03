"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
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
    
    y_train = y_train.iloc[:, 0] # Convert back to Series for SMOTE
    
    models = {
        "LogReg": ImbPipeline(steps=[
            ('preprocess', preprocessor),
            ('model', LogisticRegression(
                max_iter=params['logreg']['max_iter'],
                class_weight=params['logreg']['class_weight'],
                random_state=42
            ))
        ]),
        "RandomForest": ImbPipeline(steps=[
            ('preprocess', preprocessor),
            ('model', RandomForestClassifier(
                n_estimators=params['random_forest']['n_estimators'],
                max_depth=params['random_forest']['max_depth'],
                min_samples_split=params['random_forest']['min_samples_split'],
                class_weight=params['random_forest']['class_weight'],
                random_state=42
            ))
        ]),
        "GradientBoosting": ImbPipeline(steps=[
            ('preprocess', preprocessor),
            ('model', GradientBoostingClassifier(
                learning_rate=params['gradient_boosting']['learning_rate'],
                n_estimators=params['gradient_boosting']['n_estimators'],
                max_depth=params['gradient_boosting']['max_depth'],
                random_state=42
            ))
        ]),

        "XGBoost": ImbPipeline(steps=[
            ('preprocess', preprocessor),
            ('model', XGBClassifier(
                    n_estimators=params['xgboost']['n_estimators'],
                    learning_rate=params['xgboost']['learning_rate'],
                    max_depth=params['xgboost']['max_depth'],
                    subsample=params['xgboost']['subsample'],
                    colsample_bytree=params['xgboost']['colsample_bytree'],
                    eval_metric=params['xgboost']['eval_metric'],
                    random_state=42
            ))
        ])

    }
    
    trained_models = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        trained_models[name] = mdl
        
    return trained_models


def evaluate_models(trained_models: dict, X_test: pd.DataFrame, y_test: pd.Series, thresholds: list) -> pd.DataFrame:
    """
    Evaluates trained models across multiple probability thresholds and compiles results.
    """
    
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    # Convert back to Series
    all_results = []
    
    for thr in thresholds:
        rows = []
        for name, mdl in trained_models.items():
            # Get probabilities for the positive class (1)
            probs = mdl.predict_proba(X_test)[:, 1]
            
            # Apply the threshold to get final predictions
            y_pred = (probs >= thr).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            rows.append({
                "Threshold": thr,
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "Confusion Matrix": cm.tolist()
            })
        
        all_results.extend(rows)

    return pd.DataFrame(all_results)