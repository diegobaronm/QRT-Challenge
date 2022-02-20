import numpy as np
import pandas as pd
import os
from datetime import datetime

def dataPreProc():
    X_train = pd.read_csv('data/X_train.csv', index_col=0, sep=',')
    X_train.columns.name = 'date'

    Y_train = pd.read_csv('data/Y_train.csv', index_col=0, sep=',')
    Y_train.columns.name = 'date'

    X_train_reshape = pd.concat([ X_train.T.shift(i+1).stack(dropna=False) for i in range(250) ], 1).dropna()
    X_train_reshape.columns = pd.Index(range(1,251), name='timeLag')
    return X_train_reshape, Y_train

def getYPred(X_train_reshape, A, beta):
    Ypred = (X_train_reshape @ A @ beta).unstack().T
    return Ypred

def outputCSV(A, beta, surfix='', D=250, F=10):
    if A.shape != (D, F):
        raise ValueError('A has not the good shape')
    elif beta.shape[0] != F:
        raise ValueError('beta has not the good shape') 
    output = np.hstack( (np.hstack([A.T, beta.reshape((F, 1))])).T )
    time_str = datetime.now().strftime("%d_%m-%H_%M_")
    pd.DataFrame(output).to_csv('results/' + time_str + str(surfix) + '.csv')

def metric(df_y_true, df_y_pred):
    """ Compute metric. """
    if df_y_pred is None:  # If the y_pred has only zeroes, the metric is set to -1.
        return -1.0
    
    y_true = df_y_true.T
    y_pred = df_y_pred.T
    
    y_true = y_true.div(y_true.pow(2.0).sum(1).pow(0.5), 0)
    y_pred = y_pred.div(y_pred.pow(2.0).sum(1).pow(0.5), 0)

    mean_overlap = (y_true * y_pred).sum(1).mean()

    return mean_overlap

