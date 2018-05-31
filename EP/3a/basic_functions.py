import torch.nn.functional as F
import numpy as np
import torch
from util import randomize_in_place


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def graph1(a_np, b_np, c_np):
    """
    Computes the graph
        - x = a * c
        - y = a + b
        - f = x / y

    Computes also df/da using
        - Pytorchs's automatic differentiation (auto_grad)
        - user's implementation of the gradient (user_grad)

    :param a_np: input variable a
    :type a_np: np.ndarray(shape=(1,), dtype=float64)
    :param b_np: input variable b
    :type b_np: np.ndarray(shape=(1,), dtype=float64)
    :param c_np: input variable c
    :type c_np: np.ndarray(shape=(1,), dtype=float64)
    :return: f, auto_grad, user_grad
    :rtype: torch.DoubleTensor(shape=[1]),
            torch.DoubleTensor(shape=[1]),
            numpy.float64
    """
    
    a = torch.from_numpy(a_np)
    b = torch.from_numpy(b_np)
    c = torch.from_numpy(c_np)
    
    a.requires_grad = True
    
    x = a*c
    y = a+b
    f = x/y
    
    f.backward()
    auto_grad = a.grad
    
    df_da_x = c/y
    df_da_y = -x/(y*y)

    user_grad = (df_da_x + df_da_y).detach().numpy()
    
    return f, auto_grad, user_grad

def graph2(W_np, x_np, b_np):
    """
    Computes the graph
        - u = Wx + b
        - g = sigmoid(u)
        - f = sum(g)

    Computes also df/dW using
        - pytorchs's automatic differentiation (auto_grad)
        - user's own manual differentiation (user_grad)
        
    F.sigmoid may be useful here

    :param W_np: input variable W
    :type W_np: np.ndarray(shape=(d,d), dtype=float64)
    :param x_np: input variable x
    :type x_np: np.ndarray(shape=(d,1), dtype=float64)
    :param b_np: input variable b
    :type b_np: np.ndarray(shape=(d,1), dtype=float64)
    :return: f, auto_grad, user_grad
    :rtype: torch.DoubleTensor(shape=[1]),
            torch.DoubleTensor(shape=[d, d]),
            np.ndarray(shape=(d,d), dtype=float64)
    """
    
    W = torch.from_numpy(W_np)
    x = torch.from_numpy(x_np)
    b = torch.from_numpy(b_np)
    
    W.requires_grad = True
    
    u = torch.matmul(W, x)+b
    g = F.sigmoid(u)
    f = torch.sum(g)
    
    f.backward()
    auto_grad = W.grad    
    
    df_du = F.sigmoid(u)*(1-F.sigmoid(u))
    du_dW = np.transpose(x)
    df_dW = torch.matmul(df_du, du_dW)
    user_grad = df_dW.detach().numpy()
    
    return f, auto_grad, user_grad

def compute_cost(X, y, w):
    
    matrix = torch.matmul(X, w) - y
    matrix_T = matrix.t()
    
    J = torch.matmul(matrix_T, matrix)
    J = (1/X.shape[0])*J
    
    return J

def SGD_with_momentum(X, y, inital_w, iterations, batch_size, learning_rate, momentum):
    """
    Performs batch gradient descent optimization using momentum.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param inital_w: initial weights
    :type inital_w: np.array(shape=(d, 1))
    :param iterations: number of iterations
    :type iterations: int
    :param batch_size: size of the minibatch
    :type batch_size: int
    :param learning_rate: learning rate
    :type learning_rate: float
    :param momentum: accelerate parameter
    :type momentum: float
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    
    X_torch = torch.from_numpy(X)
    y_torch = torch.from_numpy(y)
    w_torch = torch.from_numpy(inital_w)
    
    weights_history = [inital_w]
    cost_history = [compute_cost(X_torch, y_torch, w_torch)]
    z = 0
    
    for i in range(iterations):
        w_torch.requires_grad = True
        
        index = np.random.choice(X.shape[0], batch_size, replace=False)
        sample_X = X_torch[index,:]
        sample_y = y_torch[index,:]
    
        J = compute_cost(sample_X, sample_y, w_torch)
        J.backward()
        grad = w_torch.grad
        
        z = momentum*z + grad
        
        w_np = w_torch.detach().numpy() - learning_rate*z.detach().numpy()
        w_torch = torch.from_numpy(w_np)
                
        weights_history.append(w_np)
        cost_history.append(compute_cost(X_torch, y_torch, w_torch))
    
    return w_np, weights_history, cost_history
    