from sklearn.base import BaseEstimator, TransformerMixin
from src.modules.outlier_detector import detect_outliers
import pandas as pd

class OutlierImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, multiplier=1.5):
        """
        Parameters:
            columns (list of str): List of column names to impute for outliers. If None, all numeric columns will be used.
            multiplier (float): The multiplier to define the IQR outlier bounds.
        """
        self.columns = columns
        self.multiplier = multiplier

    def fit(self, X, y=None):
        """ Compute the mean and outlier bounds for each specified column based on the IQR method. """
        X = X.copy()
        # Determine columns to process
        if self.columns is None:
            # Select all numeric columns by default
            self.columns = X.select_dtypes(include='number').columns.tolist()

        self.means_ = {}
        self.bounds_ = {}
        for col in self.columns:
            # Compute Q1, Q3, and the IQR.
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            # Calculate lower and upper bounds.
            lower_bound = Q1 - self.multiplier * IQR
            upper_bound = Q3 + self.multiplier * IQR
            self.means_[col] = X[col].mean()
            self.bounds_[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X, y=None):
        """ Replace outliers in each specified column with the pre-computed mean value. """
        X = X.copy()
        # Upcast integer columns to float to avoid dtype conflicts
        for col in self.columns:
            if pd.api.types.is_integer_dtype(X[col].dtype):
                X[col] = X[col].astype(float)

        for col in self.columns:
            lower_bound, upper_bound = self.bounds_[col]
            mean_value = self.means_[col]
            # Replace values outside the bounds with the mean.
            X.loc[(X[col] < lower_bound) | (X[col] > upper_bound), col] = mean_value
        return X