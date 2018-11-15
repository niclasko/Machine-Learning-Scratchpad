
# Logistic regression from scratch. Training using stochastic gradient descent
# Niclas Kjäll-Ohlsson, 2018

import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return (1-sigmoid(z))*sigmoid(z)

# Just plotting the sigmoid logistic function
z = [i/10 for i in range(-101,101)]
g_z = [sigmoid(i) for i in z]
plt.xlabel("z")
plt.ylabel("1/(1+exp(-z))")
plt.plot(z, g_z)
plt.show()


# Generate a classification data set
X, y = make_classification()


# Randomize model coefficents
# Model coefficients. w[0] is bias term
w = np.array([np.random.rand()*2-1 for i in range(X.shape[1]+1)])

# Performance before training
print("MSE (before training): " +       str(np.sum(np.power(preds-y, 2))))
print("Accuracy (before training): " +       str(np.sum(y == np.round(sigmoid(w[1:].dot(X.T)-w[0])))/len(y)))


# Learning
alpha = 0.1

for i in range(200000):
    ind = int(np.random.rand()*X.shape[0])
    pred = sigmoid(w[1:].dot(X[ind,].T)-w[0])
    # Application of chain rule loss'(g(z)) = loss'(g(z))*g'(z)*x_i
    # where g(z) = 1/(1+exp(-z)), and z = w.dot(X.T)-b, and loss = (g(z)-y)^2
    delta = 2*(pred-y[ind])*(1-pred)*pred
    delta_w = delta*X[ind,]
    delta_b = delta
    w[1:] = w[1:] - alpha*delta_w
    w[0] = w[0] - alpha*delta_b
    
# Predict using model
preds = sigmoid(w[1:].dot(X.T)-w[0])
    
print("MSE (after training): " +       str(np.sum(np.power(preds-y, 2))))
print("Accuracy (after training): " +       str(np.sum(y == np.round(sigmoid(w[1:].dot(X.T)-w[0])))/len(y)))




