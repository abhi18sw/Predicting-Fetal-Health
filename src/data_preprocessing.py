import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath: str):
    """
    Load dataset and perform basic cleaning.
    - Drops duplicates
    - Fills missing numeric values with column mean
    """
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    df = df.fillna(df.mean())
    return df

def split_data(df, target_col='fetal_health', test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
