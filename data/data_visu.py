import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show_missing_values(X, verbose=True):
    """Show missing values rate per columns of dataframe X

    Args:
        X (pd.DataFrame): Input data frame
    
    Return:
        missing values rate epr columns
    """
    nb_missing = X.isna().sum()
    rate_missing = nb_missing / X.ID.nunique()
    
    if verbose:
        fig, ax = plt.subplots(figsize=(4,6))
        ax1 = ax
        ax1.set_title("Missing Values Rate")
        rate_missing.plot(kind="barh", ax=ax1)
        ax1.grid()
        plt.show()
    
    return rate_missing

def show_data_distribution(lX, 
                           labels, 
                           columns, 
                           figsize=(14, 14), 
                           nb_col=None,
                           normalize=False):
    """Show distibutions of columns of X

    Args:
        X (list): list of DataFrame
        columns (list): list of columns label we want to process
    """
    X = lX[0]
    features = [feature for feature in X.columns if feature != "COUNTRY" and feature in columns]
    if nb_col is None:
        nb_col = np.clip(len(features), 0, 6)
    nb_row = - (-len(features)//nb_col)
    if not len(labels):
        while len(labels) < len(lX):
            labels.append("X_"+str(len(labels)))
    fig, ax = plt.subplots(nb_row, nb_col, figsize=figsize)
    for i, feature in enumerate(features):
        i_col = i % nb_col
        i_row = i // nb_col
        if nb_row > 1 and nb_col > 0:
            ax1 = ax[i_row, i_col]
        elif nb_col>1:
            ax1 = ax[i_col]
        else:
            ax1 = ax

        ax1.set_title(feature)
        ax1.grid()
        for X, label in zip(lX, labels):
            X[feature].hist(bins= 30, ax=ax1, alpha=0.7, label=label, 
                            density=normalize)
        ax1.legend()

    plt.tight_layout()
    plt.show()


def show_target_distribution(ly, labels=[], normalize=False):
    """Show distibutions target data y

    Args:
        y (pd.DataFrame): considered DataFrame
    """
    if not len(labels):
        while len(labels) < len(ly):
            labels.append("target_"+str(len(labels)))
    
    fig, ax = plt.subplots()
    ax1 = ax
    for y, label in zip(ly, labels):
        y["TARGET"].hist(bins= 30, ax=ax1, alpha=0.7, label=label, density=normalize)
    ax1.legend()
    ax1.set_title("TARGET")
    plt.show()


def show_missing_days(X, verbose=True):
    """Get missing days in dataframe X

    Args:
        X (pd.DataFrame): considered input dataframe
        verbose (bool, optional): plot figure if True, nothing if not. Defaults to True.

    Returns:
        pd.DataFrame: output serie
    """
    F = X.DAY_ID.value_counts().sort_index()
    F = F.reindex(range(F.index.max()))
    F = F.fillna(0)
    F = F.value_counts()
    F /= F.sum()
    
    if verbose:
        fig, ax = plt.subplots(figsize=(4,1))
        ax1 = ax
        ax1.set_title("Number of points per day in time period")
        F.plot(kind="barh", ax=ax1)
        ax1.grid()
        plt.show()
    return F


    