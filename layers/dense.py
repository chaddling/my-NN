import numpy as np
from .base import BaseLayer

class Dense(BaseLayer):
    """
        All-to-all, feedforward neural network layer:
        - size (int): specifies number of neurons in the layer = number of output units
        - input_size (int): optional argument, if specified, creates an input layer with this
          number of units

        kwargs:
        - activation_type (str): specifies the activation function used for this layer
    """

    def __init__(self,size, input_size = None, **kwargs):
        BaseLayer.__init__(self)
        self.previous = None
        self.next = None

        if input_size != None:
            self.input_size = input_size
        else:
            self.input_size = BaseLayer.last_layer_size
        self.size = size

        self.bias = np.zeros(size)
        self.dbias = np.zeros(size)
        self.units = np.zeros(size)
        self.output = np.zeros(size)

        limit = np.sqrt(6.0 / (self.size + self.input_size))

        self.weights = np.random.uniform(low=-limit, high=limit, size=(self.size, self.input_size))
        self.gradient = np.zeros((size, self.input_size))

        for key, val in kwargs.items():
            if key=='activation_type':
                activation_type = val
                self.activation_func = self.activation_func_dict[activation_type]
                self.activation_func_deriv = self.activation_deriv_dict[activation_type]

        BaseLayer.last_layer_size = self.size
