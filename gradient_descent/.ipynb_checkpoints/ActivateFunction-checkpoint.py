import numpy as np

def identity(x):
    return x

def ReLU(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return exp(np.maximum(x, 0))/(1+exp(np.maximum(x, 0)))