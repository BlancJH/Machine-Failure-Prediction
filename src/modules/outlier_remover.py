from sklearn.base import BaseEstimator, TransformerMixin
from src.modules.outlier_detector import detect_outliers
import pandas as pd

class OutlierRemover(TransformerMixin, BaseEstimator):
    """
    Scikit-learn transformer that removes outlier rows from a DataFrame
    based on the Interquartile Range (IQR) method.

    Parameters:
        multiplier (float): Factor to multiply the IQR for computing bounds.
                            Defaults to 1.5.
    """
    def __init__(self, multiplier=1.5):
        """
        Initialize the OutlierRemover.

        Args:
            multiplier (float, optional): The IQR multiplier for determining
                lower and upper bounds. Defaults to 1.5.
        """
        self.multiplier = multiplier

    def fit(self, X, y=None):
        """
        Fit method - no fitting required for outlier removal.

        Args:
            X (DataFrame or array-like): Input data.
            y (ignored): Not used, present for API consistency.

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transform method to remove outlier rows from the dataset.

        This method applies the IQR-based outlier detection to all numeric
        columns and drops any rows containing outliers.

        Args:
            X (pandas.DataFrame or array-like): The input data to clean.

        Returns:
            pandas.DataFrame or numpy.ndarray: Cleaned data with outliers removed.
                Returns the same type as the input.
        """
        # Convert to DataFrame if needed
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Detect outliers
        outliers_info = detect_outliers(df, self.multiplier)

        # Collect all indices to drop
        idx_to_drop = set()
        for info in outliers_info.values():
            idx_to_drop |= set(info['Outliers'].index)

        # Drop outlier rows
        df_clean = df.drop(index=idx_to_drop)

        # Return the cleaned data in the same format as input
        if isinstance(X, pd.DataFrame):
            return df_clean
        else:
            return df_clean.values
    