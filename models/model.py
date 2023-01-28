import numpy as np
import os

class Model:
    """
    Abstract class for all used models to be implemented.
    """

    def __init__(self, name="model"):
        self.name = name

    def fit(self, x_train, y_train):
        raise NotImplementedError("Training function not implemented.")

    def predict(self, x_test):
        raise NotImplementedError("Prediction function not implemented.")

    def __repr__(self):
        return self.name

    def submit(self, dataset, name="dummy", dir_name="challenge_data"):
        X_test = dataset.X_test
        test_IDs = dataset.get_test_IDs()
        y_test = test_IDs[["ID"]]
        y_test["TARGET"] = self.predict(X_test)
        name = name + "csv"
        dir_path = os.path.join(dir_name, "submissions")
        y_test.to_csv(os.path.join(dir_path, name), index=False)
        
        return y_test