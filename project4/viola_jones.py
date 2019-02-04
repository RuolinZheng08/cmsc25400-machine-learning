#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

class WeakLearner(object):
  '''A weak learner has a threshold, a parity and an optional weight'''
  def __init__(self, threshold, parity):
    self.threshold = threshold
    self.parity = parity
    self.weight = None

def load_image(img):
  '''Return a numpy array of a single image converted to grayscale'''
  return np.asarray(Image.open(img).convert('L'))

def get_image(path):
  '''Return the full path of images in a given directory'''
  return [os.path.join(path, f) for f in os.listdir(path)
  if not f.startswith('.')]

def load_data(faces_dir, background_dir):
  '''Return a N * 64 * 64 matrix for images and a N * 1 matrix for labels'''
  imgs = [load_image(img) for img in get_image(faces_dir) + \
  get_image(background_dir)]
  data = np.stack(imgs)
  N = data.shape[0] // 2
  labels = np.concatenate((np.ones(N, dtype='int'), 
    -np.ones(N, dtype='int')))
  return data, labels

def integral_image(arr):
  '''Compute and return the integral area representation of a matrix'''
  rows, cols = len(arr), len(arr[0])
  ret = np.empty((rows, cols), dtype='int')
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

def compute_integral_image(imgs):
  '''Compute the integral image representation for given images'''
  integrals = [integral_image(img) for img in imgs]
  return np.stack(integrals)

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
  for row in range(N):
    for col in range(N):
      if row + h <= N and col + w <= N:
        val = coord_func(row, col, h, w)
        ret.append(val)
  return ret

def feature_list(N):
  '''Return a list of features, each as (darktl, darkbr, lighttl, lightbr)'''
  vrects, hrects = [], []
  for h in range(1, N + 1, 1):
    for w in range(2, N + 1, 2):
      vshape, hshape = (w, h), (h, w)
      vrects.extend(two_rect(N, vshape, coord_vrect))
      hrects.extend(two_rect(N, hshape, coord_hrect))
  return vrects + hrects

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
  '''Compute feature evaluations given feature index'''
  features = []
  darktl, darkbr, lighttl, lightbr = feat_lst[feat_idx]
  for i in range(int_img_rep.shape[0]):
    dark = average_pixel_intensity(int_img_rep[i], darktl, darkbr)
    light = average_pixel_intensity(int_img_rep[i], lighttl, lightbr)
    features.append(dark - light)
  return np.asarray(features)

def sort_by_features(computed_features, labels, weights):
  idx = np.argsort(computed_features)
  return computed_features[idx], labels[idx], weights[idx]

def opt_theta_p(computed_features, labels, weights):
  computed_features, labels, weights = sort_by_features(computed_features, labels, weights)
  length = computed_features.shape[0]
  error_lst = np.empty(length)
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

def eval_learner(computed_features, learner):
  res = learner.parity * (computed_features - learner.threshold)
  for i in range(res.shape[0]):
    res[i] = 1 if res[i] >= 0 else -1
  return res.astype('int')

def error_rate(int_img_rep, feat_lst, weights, feat_idx, p, theta):
  pass

def opt_weaklearner(int_img_rep, weights, feat_lst):
  pass

def update_weights(weights, error_rate, y_pred, y_true):
  pass

def adaboost():
  '''The AdaBoost Algorithm'''
  weights = np.zeros(M)
  learners = []
  learner_weights = []

  for i in range(M):
    weights[i] = 1 / M
  for t in range(T):
    learner = opt_weaklearner()
    predictions = eval_learner()
    learners.append(opt_weaklearner())

    error = error_rate()
    learner_weight = compute_learner_weight(error)
    learner_weights.append(learner_weight)
    normalization = compute_normalization(error)

    weights = update_weights(weights, error, labels, predictions)

  detector = np.multiply(learners, learner_weights)
  return detector

def main():
  pass

if __name__ == '__main__':
  main()