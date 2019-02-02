#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

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
  ret = np.empty((rows, cols))
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

def compute_feature(int_img_rep, feat_lst, feat_idx):
  pass

def opt_p_theta(int_img_rep, feat_lst, weights, feat_idx):
  pass

def eval_learner(int_img_rep, feat_lst, feat_idx, p, theta):
  pass

def error_rate(int_img_rep, feat_lst, weights, feat_idx, p, theta):
  pass

def opt_weaklearner(int_img_rep, weights, feat_lst):
  pass

def update_weights(weights, error_rate, y_pred, y_true):
  pass

def main():
  pass

if __name__ == '__main__':
  main()