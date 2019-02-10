#!/usr/bin/env python3

import numpy as np
import math
from PIL import Image
import os

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
  '''Compute one feature of the two vertical rectangles'''
  return ((row, col), (row + h // 2 - 1, col + w - 1),
    (row + h // 2, col), (row + h - 1, col + w - 1))

def coord_hrect(row, col, h, w):
  '''Compute one feature of the two horizontal rectangles'''
  return ((row, col + w // 2), (row + h - 1, col + w - 1), 
    (row, col), (row + h - 1, col + w // 2 - 1))

def two_rect_feature(N, shape, step, coord_func):
  '''Compute a list of feature for the given shape and coord_func'''
  h, w = shape
  ret = []
  for row in range(0, N, step):
    for col in range(0, N, step):
      if row + h <= N and col + w <= N:
        val = coord_func(row, col, h, w)
        ret.append(val)
  return ret

def feature_list(N, base=(4, 8), step=4):
  '''Return a list of features, each as (darktl, darkbr, lighttl, lightbr)'''
  vrects, hrects = [], []
  hstep, wstep = base
  for h in range(hstep, N + 1, hstep):
    for w in range(wstep, N + 1, wstep):
      vshape, hshape = (w, h), (h, w)
      vrects.extend(two_rect_feature(N, vshape, step, coord_vrect))
      hrects.extend(two_rect_feature(N, hshape, step, coord_hrect))
  feat_lst = np.asarray(vrects + hrects, dtype='int8')
  
  return feat_lst

def preprocess_data(base=(4, 8), step=4):
  '''Load and preprocess data, return cached if exists'''
  print('Preprocessing data...')
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
    feat_lst = feature_list(int_img_rep.shape[1], base, step)
    np.save('cached/feat_lst.npy', feat_lst)

  N = int_img_rep.shape[0] // 2
  labels = \
  np.concatenate((np.ones(N, dtype='int8'), -np.ones(N, dtype='int8')))

  return int_img_rep, feat_lst, labels

################################################################################

def pixel_intensity(integral, tl, br):
  '''Compute the pixel intensity given the top-left and bottom-right loc'''
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
  return abcd - ab - ac + a

def compute_feature(int_img_rep, feature):
  '''Compute feature evaluations across all images given a feature'''
  features = np.zeros(int_img_rep.shape[0])
  darktl, darkbr, lighttl, lightbr = feature
  y1, x1 = darktl
  y2, x2 = darkbr
  for i in range(int_img_rep.shape[0]):
    dark = pixel_intensity(int_img_rep[i], darktl, darkbr)
    light = pixel_intensity(int_img_rep[i], lighttl, lightbr)
    features[i] = (dark - light) / ((x2 - x1 + 1) * (y2 - y1 + 1))
  return features

def sort_by_features(computed_features, labels, weights):
  ''''Sort computed_features, labels, weights according to computed_features'''
  idx = np.argsort(computed_features)
  return computed_features[idx], labels[idx], weights[idx]

def opt_theta_p(computed_features, labels, weights):
  '''Compute the optimal theta and p value'''
  computed_features, labels, weights = \
  sort_by_features(computed_features, labels, weights)
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
      error_lst[j], polarities[j] = bg + (afs - fs), 1
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

def compute_learner_weight(error):
  '''Compute the weight of the given learner based on its error'''
  return math.log((1 - error) / error) / 2

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
    hypotheses += eval_learner(computed_features, theta, p) * learner_weight

  return hypotheses

################################################################################

def adaboost(int_img_rep, feat_lst, labels, weights, max_iters=6):
  '''The AdaBoost Algorithm'''
  weighted_learners = np.empty(max_iters, dtype='object')
  for t in range(0, max_iters):
    print('\n***Running AdaBoost trial***', t + 1)
    learner = opt_weaklearner(int_img_rep, feat_lst, labels, weights)
    i, theta, p = learner
    computed_features = compute_feature(int_img_rep, feat_lst[i])
    hypotheses = eval_learner(computed_features, theta, p)
    
    error = error_rate(labels, hypotheses, weights)
    learner_weight = compute_learner_weight(error)

    weighted_learners[t] = (feat_lst[i], theta, p, learner_weight)
    weights = update_weights(weights, learner_weight, labels, hypotheses)

    false_pos_rate = false_positive_rate(labels, hypotheses)
    print('False Positive:', false_pos_rate)
    
    if false_pos_rate < 0.3 and t >= 3:
      break

  return weighted_learners[weighted_learners != None], weights

def cascade(int_img_rep, feat_lst, labels, max_boosters=10):
  '''Chain several boosters from different runs of AdaBoost'''
  weights = np.ones(labels.shape[0]) / labels.shape[0]
  boosters = np.empty(max_boosters, dtype='object')
  for i in range(max_boosters):
    print('\n\n---Running Cascade trial---', i + 1)
    print('Data Size:', labels.shape[0])
    booster, weights = adaboost(int_img_rep, feat_lst, labels, weights)
    hypotheses = eval_booster(int_img_rep, booster)

    print('False Negative Before Thresholding:', 
      false_negative_rate(labels, hypotheses))

    threshold = np.min(hypotheses[labels == 1])
    boosters[i] = (booster, threshold)
    hypotheses -= threshold

    print('False Positive After Thresholding:', 
      false_positive_rate(labels, hypotheses))
    
    filter_idx = (hypotheses >= 0) | (labels == 1)
    labels = labels[filter_idx]
    int_img_rep = int_img_rep[filter_idx]
    weights = weights[filter_idx]
    
    print('Remaining Data Size: {}, Faces: {}'.format(labels.shape[0],
      np.count_nonzero(labels == 1)))
  return boosters[boosters != None]

def eval_cascade(int_test_img_rep, boosters):
  '''Return N * 1 array of hypotheses for N * 64 * 64 int_img_rep'''
  faces_idx = np.arange(int_test_img_rep.shape[0])

  for i in range(boosters.shape[0]):
    booster, threshold = boosters[i]
    hypotheses = eval_booster(int_test_img_rep, booster) - threshold
    int_test_img_rep = int_test_img_rep[hypotheses >= 0]
    faces_idx = faces_idx[hypotheses >= 0]

    print('Remaining Test Points:', faces_idx.shape[0])

  return faces_idx