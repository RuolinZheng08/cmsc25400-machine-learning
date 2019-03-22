#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from neural_net import NeuralNet

def main():
  xs = np.loadtxt('TrainDigitX.csv.gz', delimiter=',')
  ys = np.loadtxt('TrainDigitY.csv', dtype='int8')
  test_xs = np.loadtxt('TestDigitX.csv.gz', delimiter=',')
  test_ys = np.loadtxt('TestDigitY.csv', dtype='int8')
  test_xs2 = np.loadtxt('TestDigitX2.csv', delimiter=',')

  lrs = [0.1, 0.5, 1, 2.5, 5]
  bs = [100, 200, 400, 1000]
  Ls = [3, 4, 5]
  Ns = [32, 64, 128, 256]

  nn = NeuralNet(1, 64, 5, 300, 100)
  errors = nn.train(xs, ys)
  test_err = nn.validate(test_xs, test_ys)
  print('Test Error for TestDigitX.csv:', test_err)
  ypreds = nn.predict(test_xs)
  ypreds2 = nn.predict(test_xs2)
  np.savetxt('TestDigitXPred.txt', ypreds, fmt='%d')
  np.savetxt('TestDigitX2Pred.txt', ypreds2, fmt='%d')

  plt.plot(errors)
  plt.show()

  # performace test for lr, b_size, n_layers, per_hidden and n_epochs
  # for lr in lrs:
  #   nn = NeuralNet(lr, 100, 4, 128, 100)
  #   nn.train(xs, ys)
  #   print(nn.validate(test_xs, test_ys))
  # for b in bs:
  #   nn = NeuralNet(1, b, 4, 64, 100)
  #   nn.train(xs, ys)
  #   print(nn.validate(test_xs, test_ys))
  # for L in Ls:
  #   nn = NeuralNet(1, 100, L, 64, 100)
  #   nn.train(xs, ys)
  #   print(nn.validate(test_xs, test_ys))
  # for N in Ns:
  #   nn = NeuralNet(1, 100, 4, N, 100)
  #   nn.train(xs, ys)
  #   print(nn.validate(test_xs, test_ys))
  # nn = NeuralNet(1, 100, 4, 128, 100)
  # for e in range(3):    
  #   nn.train(xs, ys)
  #   print(nn.validate(test_xs, test_ys))

  # plt.gray()
  # fig = plt.figure()
  # for i in range(20):
  #   ind = np.random.randint(0, test_xs2.shape[0])
  #   ax = fig.add_subplot(2, 10, i + 1)
  #   ax.set_title(str(ypreds2[i]))
  #   ax.axis('off')
  #   plt.imshow(test_xs2[ind].reshape(28, 28))
  # plt.show()

if __name__ == '__main__':
  main()