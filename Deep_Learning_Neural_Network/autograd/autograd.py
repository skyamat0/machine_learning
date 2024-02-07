import numpy as np

class Variable:
    def __init__(self, value) -> None:
        self.value = value
        self.grad = None

class Function:
    def __call__(self, input):
        x = input.value
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, *kwargs):
        raise NotImplementedError


class Polynominal(Function):
    def __init__(self, n):
        self.n = n

    def forward(self, x):
        return x ** self.n


if __name__ == "__main__":
    poly = Polynominal(3)
    x = Variable(5)
    y = poly(x)
    print(y.value)
    
