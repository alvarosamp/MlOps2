from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import os 
import sys
from prediction_model.config import config 
import numpy as np

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import numpy as np

#Classe que vamos usar para excluir colunas
class DropColumns(BaseEstimator, TransformerMixin):
    """Class to drop columns from a DataFrame."""
    def __init__(self, variables_to_drop = None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y = None):
        return self
    

    def transform(self, X):
        return X.drop(columns=self.variables_to_drop, axis=1)
    
#Domain Processing
class DomainProcessing(BaseEstimator, TransformerMixin):
    """Class to perform domain processing"""
    def __init__(self, variables_to_add = None):
        self.new_column = config.NEW_FEATURE_ADD
        self.variable_to_add = variables_to_add

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X[self.new_column] = X[self.variable_to_add].sum(axis = 1)
        return X

#Classe onde vamos aplicar label encoding
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """Class to perform label encoding"""
    def __init__(self, variables=None):
        self.variables=variables
    
    def fit(self, X,y):
        return self
    
    def transform(self,X):
        for column_name, positive_values in self.variables.items():
            X[column_name] = X[column_name].apply(
                lambda x: 1 if x.strip() in positive_values else 0
            )
        return X
    
#Realizando outra transformação
class LogTransformrs(BaseEstimator, TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X
