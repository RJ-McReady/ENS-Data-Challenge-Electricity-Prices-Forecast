import numpy as np
import pandas as pd

import sklearn.model_selection
import scipy.stats


class KFoldCrossValidation:
    def __init__(self, models, k=10):
        self._models = models
        self.nb_models = len(self._models)
        self._scores = np.zeros((k, self.nb_models), dtype=np.float64)
        self.k = k

    def run(self, x, y, verbose=True):
        kf = sklearn.model_selection.KFold(n_splits=self.k)

        for id_fold, (train_index, test_index) in enumerate(kf.split(x)):
            if verbose:
                print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = x.loc[train_index, :], x.loc[test_index, :]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            for id_model, model in enumerate(self._models):
                model.fit(x_train, y_train)
                y_hat = model.predict(x_test)
                score = scipy.stats.spearmanr(y_hat, y_test).correlation
                self._scores[id_fold, id_model] = score

    @property
    def scores(self):
        index = [f"fold_{fold}" for fold in range(self.k)]
        index = pd.Index(index, name="fold_id")
        columns = [model.name for model in self._models]
        columns = pd.Index(columns, name="model")
        scores = pd.DataFrame(self._scores, index=index, columns=columns)
        return scores


if __name__ == "__main__":
    pass