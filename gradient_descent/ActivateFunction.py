import numpy as np

def identity(x):
    return x

def ReLU(x):
    return np.maximum(x, 0)