#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def predict(weight, point):
  '''Predict 1 if dot product >= 0 else -1'''
  if np.dot(weight, point) >= 0:
    return 1
  else:
    return -1

def perceptron(data, labels, weight, mistakes=None):
  '''The Perceptron Algorithm'''
  error = 0 if not mistakes else mistakes[-1]
  for t in range(data.shape[0]):
    prediction = predict(weight, data[t])
    if prediction == 1 and labels[t] == -1:
      weight = weight - data[t]
      error += 1
    elif prediction == -1 and labels[t] == 1:
      weight = weight + data[t]
      error += 1
    if mistakes is not None:
      mistakes.append(error)
  return weight

def validate(holdout, hlabels, weight):
  error = 0
  for t in range(holdout.shape[0]):
    prediction = predict(weight, holdout[t])
    if prediction != hlabels[t]:
      error += 1
  return error

def kfoldsplit(data, k, i, stack_func):
  splits = np.array_split(data, k)
  holdout = splits[i]
  train = stack_func(splits[:i] + splits[i + 1:])
  return holdout, train

def main():
  data = np.loadtxt('train35.digits')
  labels = np.loadtxt('train35.labels')
  test = np.loadtxt('test35.digits')
  data = data / np.linalg.norm(data, axis=1, keepdims=True)
  test = test / np.linalg.norm(test, axis=1, keepdims=True)

  # cross validation
  errors = []
  for M in range(1, 11):
    error = 0
    for i in range(10):
      weight = np.array(np.zeros(data.shape[1]))
      holdout, train = kfoldsplit(data, 10, i, np.vstack)
      hlabels, tlabels = kfoldsplit(labels, 10, i, np.hstack)
      for _ in range(M):
        weight = perceptron(train, tlabels, weight)
      error += validate(holdout, hlabels, weight)
    errors.append(error / data.shape[0])
  M = np.argmin(errors) + 1

  # train the model by feeding in data M times
  mistakes = []
  weight = np.array(np.zeros(data.shape[1]))
  for _ in range(M):
    weight = perceptron(data, labels, weight, mistakes=mistakes)
  plt.plot(list(range(len(mistakes))), mistakes)
  plt.show()

  # predict on the test set
  predictions = []
  for t in range(test.shape[0]):
    prediction = predict(weight, test[t])
    predictions = np.append(predictions, prediction)
  with open('test35.predictions', 'wt') as f:
    for res in predictions:
      f.write(str(res) + '\n')

  # plt.gray()
  # fig = plt.figure()
  # for i in range(20):
  #   ind = np.random.randint(0, 200)
  #   ax = fig.add_subplot(2, 10, i + 1)
  #   ax.set_title('3' if predictions[ind] == 1 else '5')
  #   ax.axis('off')
  #   plt.imshow(test[ind].reshape(28, 28))
  # plt.savefig('result.png')

if __name__ == '__main__':
  main()