import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.getcwd())
from data.data_loading import load_data, load_data_IDs
import data.data_visu as viz

class Dataset():
    def __init__(self, dir_path="challenge_data", 
                       X_train_name="X_train.csv",
                       Y_train_name="y_train.csv",
                       X_test_name="X_test.csv"):

        self.dir_path = dir_path
        self.X_train_name = X_train_name
        self.Y_train_name = Y_train_name
        self.X_test_name = X_test_name
        
        self.reset_data(dir_path, 
                       X_train_name,
                       Y_train_name,
                       X_test_name)

        self.X_train_previous = self.X_train.copy()
        self.Y_train_previous = self.Y_train.copy()

    def reset_data(self, dir_path="challenge_data", 
                       X_train_name="X_train.csv",
                       Y_train_name="y_train.csv",
                       X_test_name="X_test.csv"):

        self.X_train, self.Y_train, self.X_test = load_data(dir_path, 
                                                            X_train_name,
                                                            Y_train_name, 
                                                            X_test_name)
    def update_data(self, X_train,
                          Y_train=None,
                          X_test=None,
                          save_previous=True):
        if save_previous:
            self.X_train_previous = self.X_train.copy()
            self.Y_train_previous = self.Y_train.copy()
        self.X_train, self.Y_train, self.X_test = X_train, Y_train, X_test

    def get_data(self):
        return self.X_train, self.Y_train, self.X_test

    def get_data_country(self, country="DE"):
        if self.Y_train is None and self.X_test is None:
            return self.X_train[self.X_train.COUNTRY == country], None, None
        elif self.Y_train is not None and self.X_test is None:
            return self.X_train[self.X_train.COUNTRY == country], self.Y_train[self.X_train[self.X_train["ID"]==self.Y_train["ID"]]["COUNTRY"] == country], None
        else:
            return self.X_train[self.X_train.COUNTRY == country], self.Y_train[self.X_train[self.X_train["ID"]==self.Y_train["ID"]]["COUNTRY"] == country], self.X_test[self.X_test.COUNTRY == country]

    ###Preprocessing method
    def normalize(self, norm_also_test=True):
        norm = lambda F: (F - F.mean(axis=0)) / F.std(axis=0) 
        self.X_train = norm(self.X_train)

        if norm_also_test:
            self.X_test = norm(self.X_test)

    def drop_features(self, columns=["ID", "DAY_ID", "COUNTRY"]):
        raise NotImplementedError("Drop features function not implemented.")

    @staticmethod
    def simple_preprop(X):
        F = X.drop("ID", axis=1)
        F = F.set_index(['DAY_ID', 'COUNTRY'])
        F = F.unstack(-1)
        F = F.fillna(0.)
        F = F.stack()
        F = F.reindex(X.set_index(['DAY_ID', 'COUNTRY']).index)
        F["ID"] = X.set_index(['DAY_ID', 'COUNTRY']).ID
        F = F.reset_index()
        F = F.drop(['ID', 'DAY_ID', 'COUNTRY'], axis = 1)
        F = (F - F.mean(axis=0)) / F.std(axis=0)
        
        return F
    
    def quick_preprocessing(self):
        self.X_train = self.simple_preprop(self.X_train)
        self.X_test = self.simple_preprop(self.X_test)
    
    def get_test_IDs(self):
        return load_data_IDs(dir_path=self.dir_path, file_path=self.X_test_name)

    ### Visualization methods
    def show_head(self):
        print("X train data : ")
        print(self.X_train.head())
        print(10*"=")
        if self.Y_train is not None:
            print("Y train data : ")
            print(self.Y_train.head())
            print(10*"=")
        else:
            print("NO LABELS")

    def show_missing_values(self):
        return viz.show_missing_values(self.X_train)
    
    def show_data_distribution(self, columns=None, 
                                     figsize=(14,14),
                                     nb_col=None, 
                                     compare_with_previous=False,
                                     compare_with_test=False,
                                     normalize=False):
        lX = []
        labels =[]

        lX.append(self.X_train)
        labels.append("X_train")
        if compare_with_test:
            lX.append(self.X_test)
            labels.append("X_test")

        if compare_with_previous:
            lX.append(self.X_train_previous)
            labels.append("X_train_previous")

        if columns is None:
            columns = self.X_train.columns
        viz.show_data_distribution(lX, 
                                   labels, 
                                   columns, 
                                   figsize, 
                                   nb_col,
                                   normalize)

    def show_target_distribution(self):
        if self.Y_train is None:
            print("NO LABELS !!")
            return
        viz.show_target_distribution([self.Y_train])        
    
    def show_missing_days(self):
        viz.show_missing_days(self.X_train)


if __name__ == "__main__":
    dataset = Dataset()
    dataset.show_data_distribution(columns=["DE_CONSUMPTION"], figsize=(2,2))
    dataset.show_data_distribution(columns=["DE_FR_EXCHANGE", "DE_CONSUMPTION"],
                                   figsize=(9,6), normalize=True)
    dataset.show_head()
    dataset.show_target_distribution()
    dataset.show_missing_values()
    dataset.show_missing_days()
