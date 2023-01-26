import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.getcwd())

import sklearn.linear_model

from models.model import *

class LinearRegressionModel(Model):
    def __init__(self, fit_intercept = True):
        super().__init__()
        self._fit_intercept = fit_intercept
        self.lr = sklearn.linear_model.LinearRegression(fit_intercept = fit_intercept)
        self._coef = None
        self._intercept = None

    def fit(self, x_train, y_train):
        self.lr.fit(x_train, y_train)
        if type(x_train) is pd.DataFrame:
            self._coef = pd.Series(self.lr.coef_, index = x_train.columns, dtype = np.float64)
            if self._fit_intercept:
                self._coef["Intercept"] = self.lr.intercept_
        else:
            self._coef = self.lr.coef_

    def predict(self, x_test):
        return self.lr.predict(x_test)

    @property
    def coef(self):
        return self._coef


