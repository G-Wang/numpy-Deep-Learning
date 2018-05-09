import numpy as np

from .activation import ActivationFunc

class ReLU(ActivationFunc):
    """Relu activation function

    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """forward pass

        """
        out = np.maximum(0, x)
        self.f_val = out
        return out


    @property
    def grad(self):
        """compute grad

        """
        assert self.f_val is not None
        return (self.f_val>0).astype(float)

    def backward(self, grad):
        """Compute gradient

        Relu's gradient is 1 if the input is >0, else gradient is 0.
        This means given the upstream gradient grad, we simply threshold it
        by checking whether the corresponding forward pass was >0 or not

        """
        g = grad*self.grad
        self.f_val = None
        return g
        