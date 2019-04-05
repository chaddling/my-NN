import numpy as np

class simpleNN:
    """
        A "model" class that keeps track of all the layers of the network,
        contains some important APIs:

        - add_layer: takes an instantiated layer object and adds
        it to the model.

        - fit_predict: takes as input, the training data set, takes
        a fraction f_train of the set for training. minibatches of size
        minibatch_size are created using the training data and after each
        epoch the loss/accuracy for the training and test data sets are both
        calculated
    """
    def __init__(self):
        self.layers_list = []

    def add_layer(self, layer):
        self.layers_list.append(layer)
        current_layer = len(self.layers_list) - 1
        if current_layer > 0:
            layer.previous = self.layers_list[current_layer - 1]
            layer.previous.next = self.layers_list[current_layer]


    def fit_predict(self, data, target, f_train, minibatch_size, eta, l2, epochs):

        indices = np.arange(len(data))
        n_train = int(f_train*len(indices))
        n_minibatch = int(n_train / minibatch_size)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_data = data[train_indices]
        train_labels = target[train_indices]

        test_data = data[test_indices]
        test_labels = target[test_indices]

        training_set = self.make_minibatch(train_data, train_labels, n_minibatch)

        for ep in range(epochs):
            output = []
            for i in range(n_minibatch):
                for x, t in training_set[i]:
                    self.forward_prop(x)
                    self.backward_prop(x, t)

                self.update(eta, l2)

            cost, accuracy = self.validate(train_data, train_labels)
            print("train_loss: {}, train_acc: {}".format(cost, accuracy))
            cost, accuracy = self.validate(test_data, test_labels)
            print("test_loss: {}, test_acc: {}".format(cost, accuracy))
            print("training completed for {} epoch(s)".format(ep+1))

    def validate(self, data, target):
        size = len(data)
        test_set = [[data[i], target[i]] for i in range(size)]
        prediction = 0
        cost = 0

        for x, t in test_set:
            output = self.forward_prop(x)
            prediction += output.dot(t)
            cost += np.sum(-t*np.log(output) - (1-t)*np.log(1-output))

        accuracy = prediction / size
        return cost, accuracy

    def make_minibatch(self, data, target, n_minibatch):
        data_size = len(data)
        minibatch_size = int(data_size / n_minibatch)
        batches = [[] for i in range(n_minibatch)]

        for n in range(n_minibatch):
            for s in range(minibatch_size):
                batches[n].append([data[n*minibatch_size + s], target[n*minibatch_size + s]])

        return batches

    def forward_prop(self, data):
        for layer in self.layers_list:
            if layer.previous == None:
                layer.units = layer.weights.dot(data) + layer.bias
                layer.output = layer.activation_func(layer.units)
            else:
                layer.units = layer.weights.dot(layer.previous.output) + layer.bias
                layer.output = layer.activation_func(layer.units)
        return layer.output # the output layer

    def backward_prop(self, data, target):
        output_layer = len(self.layers_list) - 1
        current_layer = self.layers_list[output_layer]

        a = current_layer.units
        current_layer.delta = (current_layer.output - target)
        current_layer.dbias += current_layer.delta
        current_layer.gradient +=  np.outer(current_layer.delta, current_layer.previous.output)

        # while not at the first hidden layer
        while True:
            current_layer = current_layer.previous

            a = current_layer.units
            current_layer.delta = np.dot(current_layer.next.delta, current_layer.next.weights) * current_layer.activation_func_deriv(a)
            current_layer.dbias += current_layer.delta

            if current_layer.previous != None:
                current_layer.gradient += np.outer(current_layer.delta, current_layer.previous.output)
            else:
                current_layer.gradient += np.outer(current_layer.delta, data)
                break

    # update with sgd w/ minibatches
    # uses L2 regularization
    def update(self, eta, l2):
        for layer in self.layers_list:
            layer.weights -= eta * layer.gradient + l2 * layer.weights
            layer.bias -= eta * layer.dbias

            layer.gradient = np.zeros((layer.size, layer.input_size))
            layer.dbias = np.zeros(layer.size)
