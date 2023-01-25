import numpy as np


class Model:
    """
    Abstract class for all used models to be implemented.
    """

    def __init__(self):
        raise NotImplementedError("Model is an abstract class.")

    def fit(self, x_train, y_train):
        raise NotImplementedError("Training function not implemented.")

    def predict(self, x_test):
        raise NotImplementedError("Prediction function not implemented.")
