from sklearn.preprocessing import LabelEncoder
import pandas as pd


def preprocess_data(df):

    df = df.copy()

    for col in df.columns:

        # Check if column is object/string
        if df[col].dtype == 'object':

            # Fill missing values with mode
            df[col] = df[col].fillna(df[col].mode()[0])

            # Convert text to numbers
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        else:

            # Convert column safely to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill missing numeric values with mean
            df[col] = df[col].fillna(df[col].mean())

    return df