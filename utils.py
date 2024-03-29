from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error as mse, r2_score, precision_score, recall_score

import pandas as pd
import numpy as np


def _is_cols_subset_of_df_cols(cols, df):
    """
    Utility function checking if column(s), in 'cols' is/are a subset of the
    the columns of df
    """

    if set(cols).issubset(set(df.columns)):
        return True
    else:
        raise ValueError("Class instantiated with columns that don't appear in the dataframe")


def _is_cols_input_valid(cols):
    """Utility functionchecking thevalidity of the cols parameter,
The class 'features_engineering' and 'preprocessing' assume that the
parameter is:
- Not 'None'
- Of Type list having at list one item
- All items in the list are strings
"""
    if(
        cols is None or not isinstance(cols, list) or
        len(cols) == 0 or not all(isinstance(col, str) for col in cols)
    ):
        raise ValueError(
            "Cols should be a list of strings. Each string should correspond to "
            "a column name"
        )
    else:
        return True


class LabelEncodingColumns(BaseEstimator, TransformerMixin):
    """Label Encoding selected columns
     Apply sklearn.preprocessing.Label Encoder to 'cols'

     Attributes:
     ----------
     cols : list
         List of columns in the data to be transform
    """
    def __init__(self, cols=None):
        _is_cols_input_valid(cols)
        self.cols = cols
        self.les = {col: LabelEncoder() for col in cols}
        self._is_fitted = False

    def transform(self, df, **transform_params):
        """
        Label encoding "cols" of "df" using the fitting parameters

        :param df: Dataframe
        :param transform_params:
        :return:
        """
        if not self._is_fitted:
            raise NotFittedError("Fitting was not performed")
        _is_cols_subset_of_df_cols(self.cols, df)

        df = df.copy()
        label_enc_dict = {}
        for col in self.cols:
            label_enc_dict[col] = self.les[col].transform(df[col])

        labelenc_cols = pd.DataFrame(label_enc_dict, index=df.index)

        for col in self.cols:
            df[col] = labelenc_cols[col]
        return df

    def fit(self, df, y=None, **fit_params):
        """
        Fitting the preprocessing

        :param df: DataFrame. Data
        :param y:
        :param fit_params:
        :return:
        """
        _is_cols_subset_of_df_cols(self.cols, df)
        for col in self.cols:
            self.les[col].fit(df[col])
        self._is_fitted = True
        return self


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, strategy="mean"):
        self.cols = cols
        self.strategy = strategy

    def transform(self, df):
        X = df.copy()
        impute = Imputer(strategy=self.strategy)
        if self.cols == None:
            self.cols = list(X.columns)
        for col in self.cols:
            if X[col].dtype == np.dtype('O'):
                X[col].fillna(X[col].value_counts().index[0], inplace=True)
            else:
                X[col] = impute.fit_transform(X[[col]])
        return X

    def fit(self, *_):
        return self


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._is_fitted = False

    def transform(self, df):
        if self._is_fitted:
            return pd.get_dummies(df, drop_first=True)

    def fit(self, *_):
        self._is_fitted = True
        return self


def rmse(y_true, y_pred):
    """Root Mean Squared Error. Squared MSE, in order to have the same unit between errors and y.
    :param y_true: Y with label observed
    :param y_pred: Y with label predicted
    :return: rmse score
    """
    return np.sqrt(mse(y_true, y_pred))


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error.
    RMSLE is usually used when you don't want to penalize huge differences in the predicted and the actual values when both predicted and true values are huge numbers.
    It can be used when you want to penalize under estimates more than over estimates also.
    :param y_true: Y with label observed
    :param y_pred: Y with label predicted
    :return: rmsle score
    """
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y_true]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_pred]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


def scoring_for_kind(type_ml):
    """
    Differents scoring according to the machine learning type
    :param type_ml: 'r' or 'clf'
    :return: list of scorer for quantify the performance of the ml model
    """
    if type_ml == "r":
        return [('rmse',rmse), ('rmlse', rmsle), ('r2', r2_score)]
    else:
        return[('precision', precision_score), ('recall', recall_score)]