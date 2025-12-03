"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 1.0.0
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def clean_and_process(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df_raw.copy()

    # Fill null values in loan columns with 'unknown' 
    df_cleaned = df_cleaned.fillna({
        'Housing Loan': 'unknown',
        'Personal Loan': 'unknown'
    })

    # Normalise 'Contact Method' column values
    df_cleaned['Contact Method'] = df_cleaned['Contact Method'].replace('Cell', 'cellular')
    df_cleaned['Contact Method'] = df_cleaned['Contact Method'].replace('Telephone', 'telephone')

    # Remove unrealistic '150 years' age entries
    df_cleaned['Age'] = (
        df_cleaned['Age']
            .astype(str)
            .str.replace(' years', '')
            .astype(int)
    )
    df_cleaned = df_cleaned.drop(df_cleaned[df_cleaned['Age'] == 150].index)


    # Corrects negative entries in 'Campaign Calls'
    df_cleaned['Campaign Calls'] = df_cleaned['Campaign Calls'].astype(str)
    df_cleaned['Campaign Calls'] = df_cleaned['Campaign Calls'].str.replace('-', '', regex=False)   
    df_cleaned['Campaign Calls'] = df_cleaned['Campaign Calls'].astype(int)

    # Changes '999' to '0' days 
    df_cleaned['WasContactedBefore'] = (df_cleaned['Previous Contact Days'] != 999).astype(int)
    df_cleaned['PreviouslyContacted'] = df_cleaned['Previous Contact Days'].replace({999: 0})
    df_cleaned = df_cleaned.drop(columns=['Previous Contact Days'])

    return df_cleaned

def engineer_and_prepare(df_cleaned: pd.DataFrame) -> pd.DataFrame:

    df_prep = df_cleaned.copy()
    
    # Drop client ID
    df_prep = df_prep.drop(columns=['Client ID'])
    
    # Convert target to numeric (yes: 1, no: 0)
    df_prep['Subscription Status'] = df_prep['Subscription Status'].map({'yes': 1, 'no': 0})

    # Create Age Group (categorical) from Age
    bins = [16, 25, 35, 45, 55, 65, 120]
    labels = ['16-25', '26-35', '36-45', '46-55', '56-65', '66+']
    df_prep['Age Group'] = pd.cut(
        df_prep['Age'],
        bins=bins,
        labels=labels,
        include_lowest=True,   # include 16 in first bin
        right=True             # upper bound inclusive
    )

    # Drop original numeric Age column (since we’re using Age Group instead)
    df_prep = df_prep.drop(columns=['Age'])
    

    return df_prep

def split_data(df_prep: pd.DataFrame, data_split_options: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into features (X) and target (y), then performs a train-test split.
    """
    X = df_prep.drop(columns=['Subscription Status'])
    y = df_prep['Subscription Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_split_options['test_size'],
        random_state=data_split_options['random_state'],
        stratify=y
    )
    
    # Convert y_train and y_test to DataFrame for consistent CSV storage in Kedro
    y_train = y_train.to_frame()
    y_test = y_test.to_frame()

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, features: dict) -> tuple:
    """
    Applies One-Hot Encoding and fits the ColumnTransformer.
    """
    categorical_features = features['categorical']
    numeric_features = features['numeric']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Fit and transform
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Get feature names after OHE and pass as list for easy storage
    feature_names = preprocessor.get_feature_names_out().tolist()

    # Convert to DataFrame
    X_train_processed = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)

    return X_train_processed, X_test_processed, preprocessor, feature_names