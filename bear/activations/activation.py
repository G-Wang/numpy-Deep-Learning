import numpy as np

from abc import ABC, abstractmethod

class ActivationFunc(ABC):
    """Base activation function to be inherited by all other activation functions

    """
    def __init__(self):
        """Initialze the class and create a forward pass attribute for gradient computation

        """
        self.f_val = None

    @abstractmethod
    def forward(self, x):
        """Given input x, compute forward pass output. Apply a element/row-wise activation function on input "x"

        Args:
            x (array of shape batch_size x input_dimension):
        
        Returns:
            out (array of shape batch_size x input_dimension):

        """
        pass

    @abstractmethod
    @property
    def grad(self):
        """Compute gradient of the activation function given stored f_val

        """
        pass

    @abstractmethod
    def backward(self, grad):
        """Compute the gradient of the activation function.

        Note the previous forward pass is stored internally and will be used to compute the gradient

        Args:
            grad (array): gradient from upstream of the network

        Returns:
            grad(array): computed gradient to be passed downstream

        """
        pass