import numpy as np

def identity(x):
    return x

def ReLU(x):
    return np.maximum(x, 0)

def sigmoid(x):
    if x >= 0:
        
    return np.exp(np.maximum(x, 0))/(1 + np.exp(x))