import numpy as np

class BaseLayer:
    """
        Base layer class:
        - functions and properties of the activation of the layer
        - a class variable to keep track of how many layers have been created
    """

    last_layer_size = 0

    def __init__(self):
        self.activation_func_dict = {'sigmoid': self.sigmoid,
                                    'tanh': self.tanh,
                                    'softmax': self.softmax }

        self.activation_deriv_dict = {'sigmoid': self.dsigmoid,
                                    'tanh': self.dtanh,
                                    'softmax': self.dsoftmax }

    # activation functions and derivatives...
    def tanh(self, a):
        return np.tanh(a)

    def dtanh(self, a):
        return 1.0 - self.tanh(a)**2

    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    def dsigmoid(self, a):
        return self.sigmoid(a) * (1 - self.sigmoid(a))

    def softmax(self, a):
        return np.exp(a) / np.exp(a).sum()

    def dsoftmax(self, a):
        return self.softmax(a) * (1 - self.softmax(a))
