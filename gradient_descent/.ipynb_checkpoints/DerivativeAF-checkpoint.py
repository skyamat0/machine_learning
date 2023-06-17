import numpy as np

def deriv_identity(x):
    return 1

def deriv_relu(x):
    return (x > 0).astype(x.dtype)