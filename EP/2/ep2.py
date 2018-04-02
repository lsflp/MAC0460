# Exercício 1

def standardize(X):
    """
    Returns standardized version of the ndarray 'X'.

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: standardized array
    :rtype: np.ndarray(shape=(N, d))
    """

    X_out = (X - np.mean(X))/np.std(X)

    return X_out

# Exercício 2

def compute_cost(X, y, w):
    """
    Calculates  mean square error cost.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: cost
    :rtype: float
    """

    matrix = np.dot(X, w) - y
    matrix_T = np.transpose(matrix)
    
    J = np.dot(matrix_T, matrix)
    J = J/X.shape[0]

    return J
