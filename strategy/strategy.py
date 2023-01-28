from data.pipeline import *
from models.model import *


class Strategy:
    def __init__(self,
                 pipeline: Pipeline,
                 model: Model,
                 name=None):
        self.pipeline = pipeline
        self.model = model
        self.name = model.name if name is None else name

    def process(self, x_train, y_train, x_test):
        # train
        x_train = x_train.copy().reset_index()  # pretty important for index resets
        self.pipeline(x_train)
        self.model.fit(x_train=x_train, y_train=y_train)

        # predict
        x_test = x_test.copy().reset_index()  # pretty important for index resets
        self.pipeline(x_test)
        return self.model.predict(x_test=x_test)

    def __repr__(self):
        descr = f"""Strategy {self.name}:
---{self.pipeline.__repr__()}
---{self.model.__repr__()}"""
        return descr

    def __str__(self):
        return self.__repr__()
