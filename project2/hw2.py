#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def dist(a, b):
  '''Euclidean distance'''
  return np.linalg.norm(a - b)

def random_init(data, k):
  centroids = data[np.random.choice(data.shape[0], size=k, replace=False)]
  return centroids

def kmeanspp_init(data, k):
  centroids = np.empty((0, data.shape[1]))
  centroids = np.vstack((centroids, data[np.random.choice(data.shape[0])]))
  for i in range(1, k):
    dists = np.array([np.min(np.apply_along_axis(dist, 1, centroids, data[j]))
         for j in range(data.shape[0])])
    probs = dists / np.sum(dists)
    centroids = np.vstack((centroids, data[np.random.choice(data.shape[0], p=probs)]))
  return centroids

def one_iter(data, centroids):
  n, k = data.shape[0], centroids.shape[0]
  clusters = [np.empty((0, data.shape[1]), float)] * k
  for i in range(n):
    j_hat = np.argmin(np.apply_along_axis(dist, 1, centroids, data[i]))
    clusters[j_hat] = np.vstack((clusters[j_hat], data[i]))
  for j in range(k):
    centroids[j] = np.mean(clusters[j], axis=0)
  
  return clusters

def kmeans(data, k, init_func=random_init):
  '''
  Input: data {x_1, x_2, ..., x_n}
  Output: clusters {C_1, ..., C_k}
  '''
  centroids = init_func(data, k)
  prev_distortion = -1
  distortion = -1
  distortions = []
  while True:
    prev_distotion = distortion
    clusters = one_iter(data, centroids)
    distortion = np.sum([dist(clusters[i], centroids[i]) ** 2
                       for i in range(k)])
    distortions.append(distortion)
    if prev_distotion == distortion:
      break
  plt.plot(distortions)
  return clusters

def main():
  data = np.loadtxt('mnist_small.txt')
  data = data / 255
  k = 10
  for i in range(1, 21):
    print('Running trial', i)
    np.random.seed(i)
    clusters = kmeans(data, k, init_func=kmeanspp_init)
  plt.show()

  # plt.gray()
  # for i in range(k):
  #   plt.imshow(centroids[i].reshape(8, 8))
  #   plt.show()


if __name__ == '__main__':
  main()