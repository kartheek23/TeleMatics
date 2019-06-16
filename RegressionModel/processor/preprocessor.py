import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import dask.dataframe as dd


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist median in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].median()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

class MinMaxScaling(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X=X.copy()
        scaler = MinMaxScaler()
        X=pd.DataFrame(scaler.fit_transform(X),columns=self.variables)
        return X
