# 9297961

import numpy as np

def normal_equation_prediction(X, y):
    """
    Calculates the prediction using the normal equation method.
    You should add a new column with 1s.

    :param X: design matrix
    :type X: np.array
    :param y: regression targets
    :type y: np.array
    :return: prediction
    :rtype: np.array
    """
    
    # Adicionando os 1s
    x0 = np.ones((X.shape[0], 1))
    newX = np.c_[x0, X]
    
    X_T = np.transpose(newX)
    
    # Fazendo o cálculo
    w = np.dot(X_T, newX)
    w = np.linalg.inv(w)
    w = np.dot(w, X_T)
    w = np.dot(w, y)
    
    prediction = np.dot(newX, w)

    return prediction
