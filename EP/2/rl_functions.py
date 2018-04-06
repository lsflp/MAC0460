import numpy as np
from util import randomize_in_place


def linear_regression_prediction(X, w):
    """
    Calculates the linear regression prediction.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: prediction
    :rtype: np.array(shape=(N, 1))
    """

    return X.dot(w)


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


def compute_cost(X, y, w):
    """
    Calculates  mean square error cost.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: cost
    :rtype: float
    """

    matrix = np.dot(X, w) - y
    matrix_T = np.transpose(matrix)
    
    J = np.dot(matrix_T, matrix)
    J = np.dot(1/X.shape[0], J)

    return J


def compute_wgrad(X, y, w):
    """
    Calculates gradient of J(w) with respect to w.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: gradient
    :rtype: np.array(shape=(d,))
    """

    N = X.shape[0]

    X_T = np.transpose(X)
    y_chapeu = np.dot(X, w)

    grad = np.dot(X_T, (y_chapeu - y))
    grad = np.dot(2/N, grad)

    return grad


def batch_gradient_descent(X, y, w, learning_rate, num_iters):
    """
     Performs batch gradient descent optimization.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d,)), list, list
    """

    weights_history = [w]
    cost_history = [compute_cost(X, y, w)]

    for i in range(num_iters):
        grad = compute_wgrad(X, y, w)
        w -= np.dot(learning_rate, w)
        weights_history.append(w)
        cost_history.append(compute_cost(X, y, w))

    return w, weights_history, cost_history


def stochastic_gradient_descent(X, y, w, learning_rate, num_iters, batch_size):
    """
     Performs stochastic gradient descent optimization

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :param batch_size: size of the minibatch
    :type batch_size: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    
    weights_history = [w]
    cost_history = [compute_cost(X, y, w)]
    N = X.shape[0]
    
    for i in range(num_iters):
        index = np.random.choice(N, batch_size, replace=False)
        sample_X = X[index,:]
        sample_y = y[index,:]
        grad = compute_wgrad(sample_X, sample_y, w)
        w -= np.dot(learning_rate, grad)
        weights_history.append(w)
        cost_history.append(compute_cost(X, y, w))

    return w, weights_history, cost_history
