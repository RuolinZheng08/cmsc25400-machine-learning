#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.util.shape import view_as_windows

def extract_patches(img):
  '''Return (N - 64 + 1)^2 * 64 * 64 representation of an N * N image'''
  patches = view_as_windows(img, window_shape=(64, 64), step=32)
  num_patches = patches.shape[0] * patches.shape[1]
  patches = patches.reshape((num_patches, 64, 64))
  return patches

def draw_rect(coord):
  '''Draw a red rectangle at given coord (x, y)'''
  return Rectangle(coord, 64, 64, fill=False, edgecolor='r')

def filter_idx(faces_idx):
  '''Filter '''

def get_coord(idx, width, step):
  '''Return the coord (x, y) given the index into a flat list'''
  return idx % width * step, idx // width * step

def label_image(img, faces_idx, save=False):
  '''Label the image given predictions'''
  faces_coord = [get_coord(idx, 49, 32) for idx in faces_idx]
  fig, ax = plt.subplots(1)
  ax.imshow(img)
  ax.axis('off')
  for i in range(len(faces_coord)):
    ax.add_patch(draw_rect(faces_coord[i]))
  if save:
    plg.savefig('result.jpg', quality=100)
  plt.show()

def eval_image(int_test_img_rep, boosters):
  '''Return predicted face indices for N * 64 * 64 int_img_rep'''
  faces_idx = np.arange(int_test_img_rep.shape[0])

  for i in range(boosters.shape[0]):
    booster, threshold = boosters[i]
    hypotheses = eval_booster(int_test_img_rep, booster) - threshold
    int_test_img_rep = int_test_img_rep[hypotheses >= 0]
    faces_idx = faces_idx[hypotheses >= 0]

    print('Remaining Test Points:', faces_idx.shape[0])

  return faces_idx