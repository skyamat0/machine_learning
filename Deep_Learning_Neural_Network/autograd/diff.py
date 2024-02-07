import numpy as np

class Df:
    def poly(self, x, n):
        if n == 1:
            return x
        return (n-1) * (x ** (n-1))
    
    def exp(self, x):
        return np.exp(x)
    
    def log(self, x):
        x = np.clip(x, 1e-10, 1e+10)
        return 1/x
    
    def const(self, x):
        return 0
    
    def exp_a(self, x):
        return np.log(a) * (a ** x)
    
    def sin(self, x):
        return np.cos(x)

    def cos(self, x):
        return -np.sin(x)
    
    def frac_x(self, x):
        return -1 / (x**2)
