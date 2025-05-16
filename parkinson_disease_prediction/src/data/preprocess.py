"""
This module contains a function for preprocessing data by checking and removing collinear features.
Functions:
    check_collinearity(df: pd.DataFrame) -> pd.DataFrame:
        Identifies and retains non-collinear features from the input DataFrame. 
        Features with a correlation coefficient greater than 0.7 are considered collinear 
        and are removed, except for one representative feature from each correlated group.
    A DataFrame containing only the non-collinear features.
"""

# importing the neccessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


def check_collinearity(df: pd.DataFrame) -> pd.DataFrame:
    """The function checks collinearity between the features"""

    df = df.groupby('id').mean().reset_index()
    df.drop('id', axis=1, inplace=True)

    columns = list(df.columns)
    columns_to_keep = ['class'] if 'class' in columns else []

    for col in columns:
        if col == 'class':
            continue
        is_correlated = False
        for kept in columns_to_keep:
            if kept == 'class':
                continue
            if abs(df[col].corr(df[kept])) > 0.7:
                is_correlated = True
                break
        if not is_correlated:
            columns_to_keep.append(col)

    return df[columns_to_keep]


# function to retrieve the features
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the top k features from a DataFrame using the specified statistical scoring method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column. Default is "class".
        k (int): Number of top features to select. Default is 30.
        score_func (str): Scoring function to use: 'chi2', 'f_classif', or 'mutual_info'. Default is 'chi2'.

    Returns:
        pd.DataFrame: A new DataFrame with the selected features and the target column.

    Raises:
        ValueError: If the scoring function name is invalid or if the target column is missing.
    """

    X = df.drop("class", axis=1)
    X_norm = MinMaxScaler().fit_transform(X)

    selector = SelectKBest(chi2, k=30)
    selector.fit(X_norm, df['class'])
    filtered_columns = selector.get_support()
    filtered_data = X.loc[:, filtered_columns]
    filtered_data['class'] = df['class']
    df = filtered_data
    
    return df


# function to split the data for training
def data_split(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    features = df.drop('class', axis=1)
    target = df['class']

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test
