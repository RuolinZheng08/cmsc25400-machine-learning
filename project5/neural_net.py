#!/usr/bin/env python3

import math
import numpy as np

def sigmoid(x):
  '''The sigmoid transfer function'''
  return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
  '''The derivative of the sigmoid function with respect to x'''
  return sigmoid(x) * (1 - sigmoid(x))

def softmax(yprobs):
  '''Calculate normalized softmax probabilities for the y labels'''
  exps = np.exp(yprobs)
  norms = np.sum(exps, axis=1)
  return exps / norms[:, None]

def shuffle(xs, ys):
  '''Shuffle the data and labels'''
  indices = np.arange(xs.shape[0])
  np.random.shuffle(indices)
  return xs[indices], ys[indices]

class NeuralNet():
  '''
  The Neural Net is associated with a learning rate, batch size, total number
  of layers, number of neurons per hidden layer, and number of epochs
  '''
  def __init__(self, lr, b_size, n_layers, per_hidden, n_epochs):
    self.lr = lr
    self.b_size = b_size
    self.n_layers = n_layers
    self.per_hidden = per_hidden
    self.n_epochs = n_epochs
    self.weights = None

  def init_weights(self, n_in, n_out):
    '''Initialize the weights for all layers from a normal distribution'''
    weights = [None] * self.n_layers
    weights[1] = np.random.normal(size=(self.per_hidden, n_in + 1))
    for l in range(2, self.n_layers - 1):
      weights[l] = np.random.normal(size=(self.per_hidden, self.per_hidden))
    weights[-1] = np.random.normal(size=(n_out, self.per_hidden))
    self.weights = weights

  def forward(self, batch_xs):
    '''Forward propagation on a given batch, return activations and zs'''
    alist, zlist = [None] * self.n_layers, [None] * self.n_layers
    alist[0] = np.append(batch_xs, np.ones((batch_xs.shape[0], 1)), axis=1)
    for l in range(1, self.n_layers - 1):
      zlist[l] = np.dot(alist[l - 1], self.weights[l].T)
      alist[l] = sigmoid(zlist[l])
    zlist[-1] = np.dot(alist[-2], self.weights[-1].T)
    alist[-1] = softmax(zlist[-1])
    return alist, zlist

  def backward(self, batch_ys, alist, zlist, yprobs):
    '''Backward propagation, return gradients averaged over the given batch'''
    dlist, gradients = [None] * self.n_layers, [None] * self.n_layers
    dlist[-1] = np.copy(yprobs)
    dlist[-1][np.arange(batch_ys.shape[0]), batch_ys] -= 1
    gradients[-1] = np.dot(dlist[-1].T, alist[-2])
    for l in range(self.n_layers - 2, 0, -1):
      dlist[l] = np.dot(dlist[l + 1], self.weights[l + 1]) * d_sigmoid(zlist[l])
      gradients[l] = np.dot(dlist[l].T, alist[l - 1])
    return [g / batch_ys.shape[0] if g is not None else None
            for g in gradients]

  def train(self, xs, ys):
    '''Set aside a holdout set and train the remaining dataset'''
    h_size = 10000
    holdout_xs, holdout_ys = xs[:h_size], ys[:h_size]
    train_xs, train_ys = xs[h_size:], ys[h_size:]
    errors, prev_error = [], 1

    if self.weights is None:
      self.init_weights(train_xs.shape[1], 10)

    for e in range(1, self.n_epochs + 1):
      train_xs, train_ys = shuffle(train_xs, train_ys)

      for b in range(0, train_xs.shape[0], self.b_size):
        batch_xs = train_xs[b:b + self.b_size]
        batch_ys = train_ys[b:b + self.b_size]

        alist, zlist = self.forward(batch_xs)
        yprobs = alist[-1]
        loss = -np.log10(yprobs[np.arange(yprobs.shape[0]), batch_ys])
        gradients = self.backward(batch_ys, alist, zlist, yprobs)
        self.weights = [w - self.lr * g if w is not None else None 
        for w, g in zip(self.weights, gradients)]

      error = self.validate(holdout_xs, holdout_ys)
      errors.append(error)
      if error > prev_error + 0.03:
        break
      prev_error = error
      if e % 20 == 0:
        print('Epoch {}, error {}'.format(e, error))

    return errors

  def predict(self, test_xs):
    '''Return an array of integer y label prediction'''
    alist, zlist = self.forward(test_xs)
    yprobs = alist[-1]
    return np.argmax(yprobs, axis=1)

  def validate(self, holdout_xs, holdout_ys):
    '''Compute the error again a holdout set or a test set'''
    alist, zlist = self.forward(holdout_xs)
    yprobs = alist[-1]
    yhats = np.argmax(yprobs, axis=1)
    return np.count_nonzero(yhats[yhats != holdout_ys]) / holdout_xs.shape[0]