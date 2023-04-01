import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    return (X @ Y.T + c)**p
    



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    # print("X, Y = ", X, ",",  Y)
    XTX = np.mat([np.dot(row, row) for row in X]).T
    YTY = np.mat([np.dot(row, row) for row in Y]).T
    # print("XTX, YTY = ", XTX, ",", YTY)
    XTX_matrix = np.repeat(XTX, Y.shape[0], axis=1)
    YTY_matrix = np.repeat(YTY, X.shape[0], axis=1).T
    # print("XTX_matrix, YTY_matrix = ", XTX_matrix, ",", YTY_matrix)
    K = np.asarray((XTX_matrix + YTY_matrix - 2 * (X @ Y.T)), dtype='float64')
    # print("K = ", K)
    K *= - gamma
    return np.exp(K, K)
    
