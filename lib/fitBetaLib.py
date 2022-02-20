import numpy as np

def fitBetaLinear(A, X_train_reshape, Y_train):
    predictors = X_train_reshape @ A
    targets = Y_train.T.stack()
    beta = np.linalg.inv(predictors.T @ predictors) @ predictors.T @ targets
    return beta.to_numpy()

def fitBetaNN(A, X_train_reshape, Y_train):
    pass