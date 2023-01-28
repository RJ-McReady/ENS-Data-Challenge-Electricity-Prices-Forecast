from data.pre_processing import *

import utils.formatting


class Pipeline:
    def __init__(self, pre_processing_list: list[PreProcessing]):
        self.pre_processing_list = pre_processing_list

    def step(self, x, n_step):
        for id_step, preprocessing in enumerate(self.pre_processing_list):
            if id_step == n_step:
                break
            preprocessing(x)

    def __call__(self, x):
        return self.step(x, n_step=len(self.pre_processing_list))

    def __repr__(self):
        descr = [pp.name for pp in self.pre_processing_list]
        descr = utils.formatting.list_to_str(descr, brackets="()")
        descr = f"Pipeline{descr}"
        return descr

    def __str__(self):
        return self.__repr__()
