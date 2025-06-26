# Outlier detector function based on IQR method.

import numpy as np

def detect_outliers(df, multiplier=1.5):
    """
    Detect outliers for all numeric columns in a DataFrame using the IQR method.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        multiplier (float, optional): The factor to multiply the IQR by to determine the lower and upper bounds (1.5).

    Returns:
        dict: A dictionary where keys are numeric column names and values are dictionaries
              containing Q1, Q3, IQR, Lower Bound, Upper Bound, and the outlier values.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers = df.loc[outlier_mask, col]

        outliers_info[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outliers': outliers
        }

    return outliers_info