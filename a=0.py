import numpy as np


l = [1,2,3,4,5,7]
w = np.zeros(6)
w+=np.array(l)
w[0]=999


a = [[1,1,1,5],[2,3,5,7]]
bias = np.ones(2).reshape([2,1])

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

ab = np.hstack((bias,a))

c = np.zeros(5).reshape([5,1])
c[0] = 1
c[2] = 1

d = np.array([9,2,3,4,5]).reshape([5,1])
print(c.shape)
print(d.shape)
q = (c*0.5)

print(q)

