from data.pre_processing import *


class Pipeline:
    def __init__(self, pre_processings):
        self.pre_processings = pre_processings

    def step(self, x, n_step):
        for id_step, preprocessing in enumerate(self.pre_processings):
            if id_step == n_step:
                break
            preprocessing(x)

    def __call__(self, x):
        return self.step(x, n_step=len(self.pre_processings))

    def __repr__(self):
        return f"Pipeline({[pp.__repr__() for pp in self.pre_processings]})"

    def __str__(self):
        return self.__repr__()

