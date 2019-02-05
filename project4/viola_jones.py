#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import os
import time

def timeit(method):
  '''Function decorator for timing'''
  def timed(*args, **kw):
    print('Entering', method.__name__)
    start = time.time()
    result = method(*args, **kw)
    end = time.time()
    print(method.__name__, end - start)
    return result
  return timed

################################################################################

def load_image(img):
  '''Return a numpy array of a single image converted to grayscale'''
  return np.asarray(Image.open(img).convert('L'))

def get_image(path):
  '''Return the full path of images in a given directory'''
  return [os.path.join(path, f) for f in os.listdir(path)
  if not f.startswith('.')]

@timeit
def load_data(faces_dir, background_dir):
  '''Return a N * 64 * 64 matrix of face and background images'''
  imgs = [load_image(img) for img in get_image(faces_dir) + \
  get_image(background_dir)]
  data = np.stack(imgs)
  np.save('cached/data.npy', data)
  return data

def integral_image(arr):
  '''Compute and return the integral area representation of a matrix'''
  rows, cols = arr.shape
  ret = np.zeros((rows, cols), dtype='int32')
  for row in range(rows):
    for col in range(cols):
      if row == 0 and col == 0:
        ret[row][col] = arr[row][col]
      elif row == 0:
        ret[row][col] = ret[row][col - 1] + arr[row][col]
      elif col == 0:
        ret[row][col] = ret[row - 1][col] + arr[row][col]
      else:
        ret[row][col] = ret[row][col - 1] + \
        ret[row - 1][col] + arr[row][col] - ret[row - 1][col - 1]
  return ret

@timeit
def compute_integral_image(imgs):
  '''Compute the integral image representation for given images'''
  integrals = [integral_image(img) for img in imgs]
  int_img_rep = np.stack(integrals)
  np.save('cached/int_img_rep.npy', int_img_rep)
  return int_img_rep

def coord_vrect(row, col, h, w):
  '''Compute one feature of the two vertical rectangles'''
  return ((row, col), (row + h // 2 - 1, col + w - 1),
    (row + h // 2, col), (row + h - 1, col + w - 1))

def coord_hrect(row, col, h, w):
  '''Compute one feature of the two horizontal rectangles'''
  return ((row, col + w // 2), (row + h - 1, col + w - 1), 
    (row, col), (row + h - 1, col + w // 2 - 1))

def two_rect_feature(N, shape, coord_func):
  '''Compute a list of feature for the given shape and coord_func'''
  h, w = shape
  ret = []
  for row in range(0, N, 4):
    for col in range(0, N, 4):
      if row + h <= N and col + w <= N:
        val = coord_func(row, col, h, w)
        ret.append(val)
  return ret

@timeit
def feature_list(N):
  '''Return a list of features, each as (darktl, darkbr, lighttl, lightbr)'''
  vrects, hrects = [], []
  for h in range(4, N + 1, 4):
    for w in range(8, N + 1, 8):
      vshape, hshape = (w, h), (h, w)
      vrects.extend(two_rect_feature(N, vshape, coord_vrect))
      hrects.extend(two_rect_feature(N, hshape, coord_hrect))
  feat_lst = np.asarray(vrects + hrects, dtype='int8')
  np.save('cached/feat_lst.npy', feat_lst)
  return feat_lst

def preprocess_data():
  '''Load and preprocess data, return cached if exists'''
  int_img_rep, feat_lst = None, None
  if not os.path.exists('cached'):
    os.mkdir('cached')
  if os.path.exists('cached/int_img_rep.npy'):
    int_img_rep = np.load('cached/int_img_rep.npy')
  else:
    if os.path.exists('cached/data.npy'):
      data = np.load('cached/data.npy')
    else:
      data = load_data('faces', 'background')
    int_img_rep = compute_integral_image(data)

  if os.path.exists('cached/feat_lst.npy'):
    feat_lst = np.load('cached/feat_lst.npy')
  else:
    feat_lst = feature_list(int_img_rep.shape[1])

  N = int_img_rep.shape[0] // 2
  labels = \
  np.concatenate((np.ones(N, dtype='int8'), -np.ones(N, dtype='int8')))
  return int_img_rep, feat_lst, labels

################################################################################

def average_pixel_intensity(integral, tl, br):
  '''Compute the avg pixel intensity given the top-left and bottom-right loc'''
  y1, x1 = tl
  y2, x2 = br
  abcd, ab, ac, a = 0, 0, 0, 0
  abcd = integral[y2, x2]
  if y1 > 0:
    ab = integral[y1 - 1, x2]
  if x1 > 0:
    ac = integral[y2, x1 - 1]
  if y1 > 0 and x1 > 0:
    a = integral[y1 - 1, x1 - 1]
  return (abcd - ab - ac + a) / ((x2 - x1 + 1) * (y2 - y1 + 1))

def compute_feature(int_img_rep, feat_lst, feat_idx):
  '''Compute feature evaluations across all images given feature index'''
  features = []
  darktl, darkbr, lighttl, lightbr = feat_lst[feat_idx]
  for i in range(int_img_rep.shape[0]):
    dark = average_pixel_intensity(int_img_rep[i], darktl, darkbr)
    light = average_pixel_intensity(int_img_rep[i], lighttl, lightbr)
    features.append(dark - light)
  return np.asarray(features)

def sort_by_features(computed_features, labels, weights):
  ''''Sort computed_features, labels, weights according to computed_features'''
  idx = np.argsort(computed_features)
  return computed_features[idx], labels[idx], weights[idx]

def opt_theta_p(computed_features, labels, weights):
  '''Compute the optimal theta and p value'''
  computed_features, labels, weights = \
  sort_by_features(computed_features, labels, weights)
  length = labels.shape[0]
  error_lst = np.zeros(length)
  afs = np.sum(weights[labels == 1])
  abg = np.sum(weights[labels == -1])
  fs, bg = 0, 0
  for j in range(length):
    if labels[j] == 1:
      fs += 1 * weights[j]
    else:
      bg += 1 * weights[j]
    error_lst[j] = min(bg + (afs - fs), fs + (abg - bg))
  opt_idx = np.argmin(error_lst)
  return computed_features[opt_idx], labels[opt_idx]

def eval_learner(computed_features, theta, p):
  '''Generate hypotheses by the given learner'''
  res = p * (computed_features - theta)
  for i in range(res.shape[0]):
    res[i] = 1 if res[i] < 0 else 0
  return res.astype('int8')

def error_rate(labels, hypotheses, weights):
  '''Compute the error rate of the given learner'''
  return np.dot(weights, np.absolute(hypotheses - labels))

@timeit
def opt_weaklearner(int_img_rep, feat_lst, labels, weights):
  '''Return the optimal learner in the entire feature list'''
  num_feat = feat_lst.shape[0]
  learners = np.zeros(num_feat, dtype='object')
  learner_errors = np.zeros(num_feat)
  for i in range(num_feat):
    computed_features = compute_feature(int_img_rep, feat_lst, i)
    theta, p = opt_theta_p(computed_features, labels, weights)
    
    learners[i] = (i, theta, p)
    hypotheses = eval_learner(computed_features, theta, p)
    
    learner_errors[i] = error_rate(labels, hypotheses, weights)
  opt_idx = np.argmin(learner_errors)
  return learners[opt_idx]

def compute_learner_weight(error):
  '''Compute the weight of the given learner based on its error'''
  return math.log((1 - error) / error) / 2

def update_weights(weights, error, learner_weight, labels, hypotheses):
  '''Update the weights of the dataset'''
  norm = 2 * math.sqrt(error * (1 - error))
  return weights * np.exp(-learner_weight * labels * hypotheses)

@timeit
def adaboost(int_img_rep, feat_lst, labels, num_iter):
  '''The AdaBoost Algorithm'''
  weights = np.ones(labels.shape[0]) / labels.shape[0]
  weighted_learners = np.zeros(num_iter, dtype='object')
  for t in range(0, num_iter):
    print('Running trial', t + 1)
    learner = opt_weaklearner(int_img_rep, feat_lst, labels, weights)
    i, theta, p = learner
    computed_features = compute_feature(int_img_rep, feat_lst, i)
    hypotheses = eval_learner(computed_features, theta, p)
    
    error = error_rate(labels, hypotheses, weights)
    learner_weight = compute_learner_weight(error)

    weighted_learners[t] = (i, theta, p, learner_weight)

    weights = update_weights(weights, error, learner_weight, labels, hypotheses)

  np.save('cached/detector.npy', weighted_learners)
    
  return weighted_learners

################################################################################

def main():
  if os.path.exists('cached/detector.npy'):
    detector = np.load('cached/detector.npy')
  else:
    int_img_rep, feat_lst, labels = preprocess_data()
    detector = adaboost(int_img_rep, feat_lst, labels, 20)

if __name__ == '__main__':
  main()