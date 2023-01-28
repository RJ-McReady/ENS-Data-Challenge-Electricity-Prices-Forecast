import pandas as pd
import os

def load_data(dir_path="challenge_data", 
              X_train_name="X_train.csv",
              Y_train_name="y_train.csv",
              X_test_name="X_test.csv"):
    """Load (X, Y) train data

    Args:
        dir_path (str, optional): data directory containing train csv. Defaults 
                                  to "challenge_data".
        X_train_name (str, optional): X_train data. Defaults to "X_train.csv".
        Y_train_name (str, optional): Y_train labels. Defaults to "Y_train.csv".
    """
    X_train, Y_train, X_test = None, None, None
    if X_train_name is not None:
        X_train_path = os.path.join(dir_path, 
                                    X_train_name)
        X_train = pd.read_csv(X_train_path)
    if Y_train_name is not None:
        Y_train_path = os.path.join(dir_path, 
                                    Y_train_name)
        Y_train = pd.read_csv(Y_train_path)
    if X_test_name is not None:
        X_test_path = os.path.join(dir_path, 
                                    X_test_name)
        X_test = pd.read_csv(X_test_path)

    return X_train, Y_train, X_test

def load_data_IDs(dir_path="challenge_data", file_path="X_test.csv"):
    path = os.path.join(dir_path, 
                        file_path)
    return pd.read_csv(path)[["ID", "DAY_ID", "COUNTRY"]]

if __name__ == "__main__":
    # X_train, Y_train, X_test = load_data()
    load_data_IDs()




    