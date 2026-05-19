import pandas as pd


def detect_task(df):
    target = df.columns[-1]

    if 'date' in target.lower() or 'time' in target.lower():
        return 'time_series'

    unique_values = df[target].nunique()

    if df[target].dtype == 'object':
        return 'classification'

    if unique_values < 10:
        return 'classification'

    if unique_values > 20:
        return 'regression'

    return 'clustering'