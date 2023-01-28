import numpy as np
import pandas as pd
import os, sys

import sklearn.linear_model

from models.model import *
import matplotlib.pyplot as plt
import sklearn.linear_model

from models.model import *

sys.path.append(os.getcwd())


class LinearRegressionModel(Model):
    def __init__(self, fit_intercept=True, name="LinReg"):
        super().__init__(name=name)
        self._fit_intercept = fit_intercept
        self.lr = sklearn.linear_model.LinearRegression(fit_intercept=fit_intercept)
        self._coef = None
        self._intercept = None

    def fit(self, x_train, y_train):
        self.lr.fit(x_train, y_train)
        if type(x_train) is pd.DataFrame:
            self._coef = pd.Series(self.lr.coef_, index=x_train.columns, dtype=np.float64)
            if self._fit_intercept:
                self._coef["Intercept"] = self.lr.intercept_
        else:
            self._coef = self.lr.coef_

    def predict(self, x_test):
        return self.lr.predict(x_test)

    def plot_coef(self):
        fig, ax = plt.subplots(figsize=(6, 0.2 * len(self._coef.index)))
        ax1 = ax
        self._coef.plot(kind="barh", ax=ax1)
        ax1.grid()
        ax1.axvline(0, color="black", linestyle="--")
        return fig, ax

    @property
    def coef(self):
        return self._coef


class RidgeRegressionModel(LinearRegressionModel):
    def __init__(self, alpha=1., fit_intercept=True, name="RidgeReg"):
        super().__init__(fit_intercept=fit_intercept, name=name)
        self.alpha = alpha
        self.lr = sklearn.linear_model.Ridge(alpha=self.alpha, fit_intercept=fit_intercept)


class LassoRegressionModel(LinearRegressionModel):
    def __init__(self, alpha=1., fit_intercept=True, name="LassoReg"):
        super().__init__(fit_intercept=fit_intercept, name=name)
        self.alpha = alpha
        self.lr = sklearn.linear_model.Lasso(alpha=self.alpha, fit_intercept=fit_intercept)
