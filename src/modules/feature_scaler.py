import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler

# Define a preprocessor that scales only numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            MinMaxScaler(),
            make_column_selector(dtype_include=np.number)  
        )
    ],
    remainder='passthrough'  # leave all other columns unchanged
)
