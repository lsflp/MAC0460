import numpy as np

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

# Exercício 3

def compute_wgrad(X, y, w):
    """
    Calculates gradient of J(w) with respect to w.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: gradient
    :rtype: np.array(shape=(d, 1))
    """
    
    eps = 1e-4
    d = w.shape[0]
    grad = np.array([])
    
    for i in range(0, d):
        w_mais = np.array(w, copy=True)
        w_menos = np.array(w, copy=True)
        
        w_mais[i] += eps
        w_menos[i] -= eps
        
        J_mais = compute_cost(X, y, w_mais)
        J_menos = compute_cost(X, y, w_menos)

        grad = np.append(grad, (J_mais - J_menos)/(2*eps))
        
    grad = np.reshape(grad, (d, 1))
      
    return grad

# Exercício 4

def batch_gradient_descent(X, y, w, learning_rate, num_iters):
    """
     Performs batch gradient descent optimization.

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
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    
    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]

    for i in range(num_iters):
        grad = compute_wgrad(X, y, w)
        w -= learning_rate*grad
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X, y, w))

    return w, weights_history, cost_history 

#Exercício 5

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
    
    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]
    N = X.shape[0]
    
    for i in range(num_iters):
        index = np.random.choice(N, batch_size, replace=False)
        sample_X = X[index,:]
        sample_y = y[index,:]
        grad = compute_wgrad(sample_X, sample_y, w)
        w -= learning_rate*grad
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X, y, w))

    return w, weights_history, cost_history