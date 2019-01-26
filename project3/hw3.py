#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def predict(weight, point):
  '''Predict 1 if dot product >= 0 else -1'''
  if np.dot(weight, point) >= 0:
    return 1
  else:
    return -1

def perceptron(data, labels):
  '''The Perceptron Algorithm'''
  weight = np.array(np.zeros(data.shape[1]))
  count, mistakes = 0, []
  for t in range(data.shape[0]):
    prediction = predict(weight, data[t])
    if prediction == 1 and labels[t] == -1:
      weight = weight - data[t]
      count += 1
    elif prediction == -1 and labels[t] == 1:
      weight = weight + data[t]
      count += 1
    mistakes.append(count)
  plt.plot(list(range(data.shape[0])), mistakes)
  plt.show()
  return weight

def main():
  data = np.loadtxt('train35.digits')
  labels = np.loadtxt('train35.labels')
  test = np.loadtxt('test35.digits')
  data = data / np.linalg.norm(data, axis=1, keepdims=True)
  test = test / np.linalg.norm(test, axis=1, keepdims=True)
  weight = perceptron(data, labels)
  predictions = []
  for t in range(test.shape[0]):
    prediction = predict(weight, test[t])
    predictions = np.append(predictions, prediction)
  with open('test35.predictions', 'wt') as f:
    for res in predictions:
      f.write(str(res) + '\n')

if __name__ == '__main__':
  main()