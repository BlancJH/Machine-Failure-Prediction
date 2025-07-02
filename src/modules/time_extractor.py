from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column) -> None:
        self.column = column

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Determine columns to process
        if self.column is None:
            # Select all numeric columns by default
            self.column = 'Time'

        X[self.column] = pd.to_datetime(X[self.column])
        X['year'] = X[self.column].dt.year
        X['month'] = X[self.column].dt.month
        X['day'] = X[self.column].dt.day
        X['hour'] = X[self.column].dt.hour
        X['minute'] = X[self.column].dt.minute
        X['weekday'] = X[self.column].dt.weekday # 0=Mon,6=Sun
        X['is_weekend'] = X[self.column].dt.weekday >= 5
        
        return X.drop(columns=[self.column])