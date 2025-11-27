"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Split data into Xtrain, Xtest, ytrain, ytest
def split_data(data: pd.DataFrame, parameters: dict):
    X = data.drop(columns=[parameters["target_column"]])
    y = data[parameters["target_column"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test

#Model Training 
def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    model = LogisticRegression() #depends on the model used (JL)
    model.fit(X_train, y_train)
    return model