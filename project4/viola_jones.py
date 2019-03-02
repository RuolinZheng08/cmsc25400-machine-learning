#!/usr/bin/env python3

import numpy as np
import math
from PIL import Image
import os
import time

################################################################################

def load_image(img):
  '''Return a numpy array of a single image converted to grayscale'''
  return np.asarray(Image.open(img).convert('L'))

def get_image(path):
  '''Return the full path of images in a given directory'''
  return [os.path.join(path, f) for f in os.listdir(path)
  if not f.startswith('.')]

def load_data(faces_dir, background_dir):
  '''Return an N * 64 * 64 matrix of face and background images'''
  imgs = [load_image(img) for img in get_image(faces_dir) + \
  get_image(background_dir)]
  data = np.stack(imgs)
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

def compute_integral_image(imgs):
  '''Compute the integral image representation for given images'''
  integrals = [integral_image(img) for img in imgs]
  int_img_rep = np.stack(integrals)
  return int_img_rep

def coord_vrect(row, col, h, w):
  '''Compute one feature of the two vertical rectangles, top dark'''
  return ((row, col), (row + h // 2 - 1, col + w - 1),
    (row + h // 2, col), (row + h - 1, col + w - 1))

def coord_hrect(row, col, h, w):
  '''Compute one feature of the two horizontal rectangles, right dark'''
  return ((row, col + w // 2), (row + h - 1, col + w - 1), 
    (row, col), (row + h - 1, col + w // 2 - 1))

def coord_trect(row, col, h, w):
  '''Compute one feature of the three horizontal rectangles, middle dark'''
  return ((row, col + w // 3), (row + h - 1, col + w // 3 * 2 - 1), 
    (row, col), (row + h - 1, col + w // 2 - 1),
    (row, col + w // 3 * 2), (row + h - 1, col + w - 1))

def rect_feature(N, shape, coord_func):
  '''Compute a list of feature for the given shape and coord_func'''
  h, w = shape
  ret = []
  for row in range(0, N, 4):
    for col in range(0, N, 4):
      if row + h <= N and col + w <= N:
        val = coord_func(row, col, h, w)
        ret.append(val)
  return ret

def feature_list(N):
  '''
  Return a list of features, each as (darktl, darkbr, lighttl, lightbr)
  or (darktl, darkbr, lighttl, lightbr, lighttl2, lightbr2)
  '''
  vrects, hrects, trects = [], [], []
  for h in range(4, N + 1, 4):
    for w in range(8, N + 1, 8):
      vshape, hshape = (w, h), (h, w)
      vrects.extend(rect_feature(N, vshape, coord_vrect))
      hrects.extend(rect_feature(N, hshape, coord_hrect))
  for h in range(6, N + 1, 6):
    for w in range(9, N + 1, 9):
      trects.extend(rect_feature(N, (h, w), coord_trect))
  feat_lst = np.asarray(vrects + hrects + trects)
  
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
    np.save('cached/int_img_rep.npy', int_img_rep)

  if os.path.exists('cached/feat_lst.npy'):
    feat_lst = np.load('cached/feat_lst.npy')
  else:
    feat_lst = feature_list(int_img_rep.shape[1])
    np.save('cached/feat_lst.npy', feat_lst)

  N = int_img_rep.shape[0] // 2
  labels = \
  np.concatenate((np.ones(N, dtype='int8'), -np.ones(N, dtype='int8')))

  return int_img_rep, feat_lst, labels

################################################################################

def pixel_intensity(int_img_rep, tl, br):
  '''Compute the pixel intensity given the top-left and bottom-right loc'''
  y1, x1 = tl
  y2, x2 = br
  abcd, ab, ac, a = 0, 0, 0, 0
  abcd = int_img_rep[:, y2, x2]
  if y1 > 0:
    ab = int_img_rep[:, y1 - 1, x2]
  if x1 > 0:
    ac = int_img_rep[:, y2, x1 - 1]
  if y1 > 0 and x1 > 0:
    a = int_img_rep[:, y1 - 1, x1 - 1]
  return abcd - ab - ac + a

def compute_two_rect_feature(int_img_rep, feature):
  '''Compute (darktl, darkbr, lighttl, lightbr)'''
  darktl, darkbr, lighttl, lightbr = feature
  y1, x1 = darktl
  y2, x2 = darkbr
  dark = pixel_intensity(int_img_rep, darktl, darkbr)
  light = pixel_intensity(int_img_rep, lighttl, lightbr)
  return (dark - light) / ((x2 - x1 + 1) * (y2 - y1 + 1))

def compute_three_rect_feature(int_img_rep, feature):
  '''Compute (darktl, darkbr, lighttl, lightbr, lighttl2, lightbr2)'''
  darktl, darkbr, lighttl, lightbr, lighttl2, lightbr2 = feature
  y1, x1 = darktl
  y2, x2 = darkbr
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  dark = pixel_intensity(int_img_rep, darktl, darkbr)
  light = pixel_intensity(int_img_rep, lighttl, lightbr)
  light2 = pixel_intensity(int_img_rep, lighttl2, lightbr2)
  return dark / area - (light + light2) / (2 * area)

def compute_feature(int_img_rep, feature):
  '''Compute feature evaluations across all images given a feature'''
  if len(feature) == 4:
    return compute_two_rect_feature(int_img_rep, feature)
  else:
    return compute_three_rect_feature(int_img_rep, feature)

def opt_theta_p(computed_features, labels, weights):
  '''Compute the optimal theta and p value'''
  idx = np.argsort(computed_features)

  computed_features = computed_features[idx]
  labels = labels[idx]
  weights = weights[idx]

  length = labels.shape[0]
  error_lst, polarities = np.zeros(length), np.zeros(length, dtype='int8')
  afs = np.sum(weights[labels == 1])
  abg = np.sum(weights[labels == -1])
  fs, bg = 0, 0
  for j in range(length):
    if labels[j] == 1:
      fs += weights[j]
    else:
      bg += weights[j]
    if bg + (afs - fs) < fs + (abg - bg):
      error_lst[j], polarities[j] = bg + (afs - fs), -1
    else:
      error_lst[j], polarities[j] = fs + (abg - bg), 1
  opt_idx = np.argmin(error_lst)

  if opt_idx == length - 1:
    theta = computed_features[opt_idx]
  else:
    theta = (computed_features[opt_idx] + computed_features[opt_idx + 1]) / 2
  return theta, polarities[opt_idx]

def eval_learner(computed_features, theta, p):
  '''Return hypotheses 1 or -1 by the given learner'''
  res = p * (computed_features - theta)
  res = np.sign(res) 
  res[res == 0] = 1
  return res.astype('int8')

def error_rate(labels, hypotheses, weights):
  '''Compute the error rate of the given learner'''
  return np.dot(weights, np.absolute((hypotheses - labels) / 2))

def false_positive_rate(labels, hypotheses):
  '''Compute FPR = FP / N = FP / (FP + TN)'''
  fpos = np.count_nonzero((labels == -1) & (hypotheses >= 0))
  tneg = np.count_nonzero(labels == -1)
  return fpos / (fpos + tneg)

def false_negative_rate(labels, hypotheses):
  '''Compute FNR = FN / P = FN / (FN + TP)'''
  fneg = np.count_nonzero((labels == 1) & (hypotheses < 0))
  tpos = np.count_nonzero(labels == 1)
  return fneg / (fneg + tpos)

def opt_weaklearner(int_img_rep, feat_lst, labels, weights):
  '''Return the optimal learner in the entire feature list'''
  num_feat = feat_lst.shape[0]
  learners = np.empty(num_feat, dtype='object')
  learner_errors = np.zeros(num_feat)
  for i in range(num_feat):
    computed_features = compute_feature(int_img_rep, feat_lst[i])
    theta, p = opt_theta_p(computed_features, labels, weights)
    
    learners[i] = (i, theta, p)
    hypotheses = eval_learner(computed_features, theta, p)
    
    learner_errors[i] = error_rate(labels, hypotheses, weights)
  opt_idx = np.argmin(learner_errors)
  print('Min Error:', learner_errors[opt_idx])
  return learners[opt_idx]

def update_weights(weights, learner_weight, labels, hypotheses):
  '''Update the weights of the dataset'''
  weights = weights * np.exp(-learner_weight * labels * hypotheses)
  norm = np.sum(weights)
  return weights / norm

def eval_booster(int_img_rep, weighted_learners):
  '''Return raw values of prediction by the given set of weighted learners'''
  hypotheses = np.zeros(int_img_rep.shape[0])

  for t in range(weighted_learners.shape[0]):
    feature, theta, p, learner_weight = weighted_learners[t]
    computed_features = compute_feature(int_img_rep, feature)
    hypotheses += p * (computed_features - theta) * learner_weight
  print(hypotheses)
  return hypotheses

################################################################################

def adaboost(int_img_rep, feat_lst, labels, weights, max_iters=12):
  '''The AdaBoost Algorithm'''
  weighted_learners = np.empty(max_iters, dtype='object')
  for t in range(0, max_iters):
    print('\n***Running AdaBoost trial***', t + 1)
    learner = opt_weaklearner(int_img_rep, feat_lst, labels, weights)
    i, theta, p = learner
    computed_features = compute_feature(int_img_rep, feat_lst[i])
    hypotheses = eval_learner(computed_features, theta, p)
    
    error = error_rate(labels, hypotheses, weights)
    learner_weight = math.log((1 - error) / error) / 2

    weighted_learners[t] = (feat_lst[i], theta, p, learner_weight)
    weights = update_weights(weights, learner_weight, labels, hypotheses)

    hypotheses = eval_booster(int_img_rep, 
      weighted_learners[weighted_learners != None])
    false_pos_rate = false_positive_rate(labels, hypotheses)
    print('False Positive Of Strong Learner:', false_pos_rate)
    
    if false_pos_rate < 0.2:
      break

  return weighted_learners[weighted_learners != None], weights

def cascade(int_img_rep, feat_lst, labels, max_boosters=15):
  '''Chain several boosters from different runs of AdaBoost'''
  weights = np.ones(labels.shape[0]) / labels.shape[0]
  boosters = np.empty(max_boosters, dtype='object')
  false_pos_rate = 1

  for t in range(max_boosters):
    print('\n\n---Running Cascade trial---', t + 1)
    print('Data Size:', labels.shape[0])
    booster, weights = adaboost(int_img_rep, feat_lst, labels, weights)
    hypotheses = eval_booster(int_img_rep, booster)

    print('\n\nFalse Negative Before Thresholding:', 
      false_negative_rate(labels, hypotheses))
    print('False Positive Before Thresholding:', 
      false_positive_rate(labels, hypotheses))
    print('Threshold:', np.min(hypotheses[labels == 1]), 
      'Overall:', np.min(hypotheses))

    threshold = np.min(hypotheses[labels == 1])
    hypotheses -= threshold
    boosters[t] = (booster, threshold)
    np.save('cached/boosters.npy', boosters)

    false_pos_rate *= false_positive_rate(labels, hypotheses)
    print('False Positive Of Cascade:', false_pos_rate)
    if false_pos_rate < 0.01:
      break
    
    filter_idx = (hypotheses >= 0) | (labels == 1)
    labels = labels[filter_idx]
    int_img_rep = int_img_rep[filter_idx]
    weights = weights[filter_idx]

    if labels.shape[0] == np.count_nonzero(labels == 1):
      break
    print('Remaining Data Size: {}, Faces: {}'.format(labels.shape[0],
      np.count_nonzero(labels == 1)))

  return boosters[boosters != None]

def eval_cascade(int_img_rep, boosters):
  '''Return N * 1 array of hypotheses for N * 64 * 64 int_img_rep'''
  hypotheses = np.zeros(int_img_rep.shape[0])

  for i in range(boosters.shape[0]):
    booster, threshold = boosters[i]
    hypotheses = eval_booster(int_img_rep, booster) - threshold
    int_img_rep = int_img_rep[hypotheses >= 0]

  return hypotheses