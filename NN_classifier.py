import matplotlib.pyplot as plt
import numpy as np

from layers import Dense
from models import simpleNN

# create some example data:
# a non-linearly-separable classifcation problem in 2D
N_samples = 500
data = []
labels = []
indices = range(N_samples)

for i in indices:
    x = np.random.uniform(-1.0, 1.0, size=2)
    data.append(x,)

    if x[0]**2 + x[1]**2 < 1/4:
        labels.append(np.array([1, 0]))
    else:
        labels.append(np.array([0, 1]))

data = np.array(data)
labels = np.array(labels)

indices_disk = set(np.where(labels[:]==[1, 0])[0])
indices_disk = list(indices_disk)

indices_not_disk = [i for i in indices if i not in indices_disk]

disk = data[indices_disk]
not_disk = data[indices_not_disk]

plt.plot(disk[:,0], disk[:,1],marker='o',linestyle='')
plt.plot(not_disk[:,0], not_disk[:,1],marker='o',linestyle='')
plt.show()

# build the network
myNN = simpleNN()
myNN.add_layer(Dense(4, input_size=2, activation_type='tanh'))
myNN.add_layer(Dense(2, activation_type='softmax'))

myNN.fit_predict(data=data, target=labels, f_train=0.8, minibatch_size=50, eta=0.1, l2=0.0, epochs=100)
