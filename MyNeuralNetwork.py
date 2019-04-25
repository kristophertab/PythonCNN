import torch
import math
import numpy as np
import gzip
import matplotlib.pyplot as plt
import idx2numpy
from matplotlib import pyplot
import tensorflow as tf

image_size = 28*28

file = gzip.open("./data/FashionMNIST/FashionMNIST/fashion/train-images-idx3-ubyte.gz", 'r')
x_train = idx2numpy.convert_from_file(file).astype('float32')
file = gzip.open("./data/FashionMNIST/FashionMNIST/fashion/train-labels-idx1-ubyte.gz", 'r')
y_train = idx2numpy.convert_from_file(file).astype('int64')
file = gzip.open("./data/FashionMNIST/FashionMNIST/fashion/t10k-images-idx3-ubyte.gz", 'r')
x_test = idx2numpy.convert_from_file(file).astype('float32')
file = gzip.open("./data/FashionMNIST/FashionMNIST/fashion/t10k-labels-idx1-ubyte.gz", 'r')
y_test = idx2numpy.convert_from_file(file).astype('int64')

x_train = x_train.reshape((x_train.shape[0], image_size))
x_test = x_test.reshape((x_test.shape[0], image_size))

classes = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Shoe']

x_train, y_train, x_test, y_test = map(torch.tensor, (x_train, y_train, x_test, y_test))
print(x_train.shape)
n, c = x_train.shape

#  initializing the weights here with Xavier initialisation (by multiplying with 1/sqrt(n)
neurons = 10
weights = torch.randn(784, neurons) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(neurons, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def reLU(x):
    (x>0)

def model(xb):
    dot_result = xb @ weights + bias
    return log_softmax(dot_result)

bs = 64

x_train = x_train[:n, :]
y_train = y_train[:n]

xb = x_train[0:bs]
preds = model(xb)
print("not teached output")
print(preds[0], preds.shape)

# for i in range(10):
#     pyplot.imshow(x_train[i].reshape((28, 28)), cmap="gray")
#     plt.title(classes[y_train[i]])
#     plt.show()


# Negative Log-Likelihood
def nll(input, target):
    ret = -input[range(target.shape[0]), target]
    return ret.mean()

loss_func = nll
yb = y_train[0:bs]#.unsqueeze(0)
print("loss function")
print(loss_func(preds, yb))

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    eq = preds == preds
    f_eq = eq.float()
    ret = f_eq.mean()
    return ret
print("accuracy")
print(accuracy(preds, yb))

#teaching
lr = 0.1
epochs = 33 

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print("results of trained model: loss accuracy")
print(loss_func(model(xb), yb), accuracy(model(xb), yb))