import numpy as np
import pandas as pd


class PreProcessing:
    def __init__(self, name="pre_processing", columns=None):
        self.name = name
        self._columns = columns

    def columns(self, x):
        return x.columns if self._columns is None else pd.Index(self._columns)

    def __call__(self, x):
        raise NotImplementedError("PreProcessing is an abstract class.")

    def __repr__(self):
        return str(self.name)


class MissingValuesLast(PreProcessing):
    def __init__(self, name="missing_values_last", columns=None):
        super().__init__(name, columns)

    def __call__(self, x, default_value=0.):
        cols = self.columns(x)
        cols_indexer = pd.Index(['DAY_ID', 'COUNTRY']) \
            .append(cols).unique()
        acc = x[cols_indexer]
        acc = acc.set_index(['DAY_ID', 'COUNTRY']).unstack(-1)
        acc = acc.fillna(method="ffill")
        if default_value is None:
            pass
        else:
            acc = acc.fillna(default_value)  # for first values
        acc = acc.stack()
        acc = acc.reindex(x.set_index(['DAY_ID', 'COUNTRY']).index)
        acc = acc.reset_index()
        x.loc[:, cols] = acc.loc[:, cols]


class MissingValuesMean(PreProcessing):
    def __init__(self, name="missing_values_mean", columns=None):
        super().__init__(name, columns)

    def __call__(self, x):
        cols = self.columns(x)
        for col in cols:
            try:
                m = x[col].mean()
                x[col].fillna(m, inplace=True)
            except TypeError:
                pass


class CenteredReduced(PreProcessing):
    def __init__(self, name="centered_reduced", columns=None):
        super().__init__(name, columns)

    def __call__(self, x):
        cols = self.columns(x)
        x.loc[:, cols] = (x.loc[:, cols] - x.loc[:, cols].mean(axis=0)) / x.loc[:, cols].std(axis=0)


if __name__ == "__main__":
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, sys
    from data.data_loading import load_data

    project_root_dir = os.path.dirname(os.getcwd())
    data_dirname = os.path.join(project_root_dir, 'challenge_data')
    # read data
    X_train, y_train, X_test = load_data(data_dirname)

    from data.pre_processing import *

    X = X_test.copy()
    mv = MissingValuesLast()
    mv(X)
    X.head()
