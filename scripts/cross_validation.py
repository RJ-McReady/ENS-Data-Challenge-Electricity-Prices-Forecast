from strategy.strategy import *

import numpy as np
import pandas as pd
import sys

import sklearn.model_selection
import scipy.stats


class KFoldCrossValidation:
    def __init__(self, strategies, k=10):
        self._strategies = strategies
        self.nb_strategies = len(self._strategies)
        self._scores = np.zeros((k, self.nb_strategies), dtype=np.float64)
        self.k = k

    def run(self, x, y, verbose=True):
        kf = sklearn.model_selection.KFold(n_splits=self.k)

        for id_fold, (train_index, test_index) in enumerate(kf.split(x)):
            if verbose:
                print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = x.loc[train_index, :], x.loc[test_index, :]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            for id_strategy, strategy in enumerate(self._strategies):
                y_hat = strategy.process(x_train, y_train, x_test)
                score = scipy.stats.spearmanr(y_hat, y_test).correlation
                self._scores[id_fold, id_strategy] = score

    @property
    def scores(self):
        index = [f"fold_{fold}" for fold in range(self.k)]
        index = pd.Index(index, name="fold_id")
        columns = [strategy.name for strategy in self._strategies]
        columns = pd.Index(columns, name="strategy")
        scores = pd.DataFrame(self._scores, index=index, columns=columns)
        return scores


if __name__ == "__main__":
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, sys

    sys.path.append(os.path.dirname(os.path.abspath('')))
    from data.data_loading import load_data

    project_root_dir = os.path.dirname(os.getcwd())
    data_dirname = os.path.join(project_root_dir, 'challenge_data')
    # read data
    X_train, y_train, X_test = load_data(data_dirname)

    X_train.columns

    from strategy.strategy import *
    from models.linreg import *
    from scripts.cross_validation import *
    from itertools import product

    PP1 = Pipeline([MissingValuesMean(), DropIds(), CenteredReduced()])
    PP2 = Pipeline([MissingValuesLast(), DropIds(), CenteredReduced()])

    pipelines = [PP2, PP1]

    ridge_pen = [10 ** (i / 2) for i in range(-6, 10)]
    ridge_models = [RidgeRegressionModel(alpha=alpha, name=f"RidgeReg_{alpha:0.3g}") for alpha in ridge_pen]

    lasso_pen = [10 ** (i / 2) for i in range(-14, -1)]
    lasso_models = [LassoRegressionModel(alpha=alpha, name=f"LassoReg_{alpha:0.3g}") for alpha in lasso_pen]

    models = [LinearRegressionModel()] + ridge_models + lasso_models

    strategies = [Strategy(pipeline, model, name=f"strat{id_s}") for id_s, (pipeline, model) in
                  enumerate(product(pipelines, models))]

    CV = KFoldCrossValidation(strategies, k=5)
    CV.run(X_train, y_train.TARGET, verbose=False)