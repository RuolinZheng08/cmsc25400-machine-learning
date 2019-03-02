#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.util.shape import view_as_windows, view_as_blocks
from PIL import Image
from viola_jones import compute_integral_image, eval_cascade

def non_maximum_suppresion(imgarr, boosters):
  '''Return only the maximum hypothesis values for 128 * 128 boxes in image'''
  patches = view_as_windows(imgarr, window_shape=(128, 128), step=64)
  rows, cols = patches.shape[0], patches.shape[1]
  predictions = np.zeros((rows, cols))
  for row in range(rows):
    for col in range(cols):
      patch = view_as_blocks(np.ascontiguousarray(patches[row, col]), 
        block_shape=(64, 64))
      patch = patch.reshape((patch.shape[0] * patch.shape[1], 64, 64))

      hypotheses = eval_cascade(patch, boosters)

      predictions[row, col] = np.max(hypotheses)
  return predictions

def draw_rect(coord):
  '''Draw a red rectangle at given coord (x, y)'''
  return Rectangle(coord, 64, 64, fill=False, edgecolor='r')

def label_image(fname, boosters, outfname=None):
  '''Label the image with non-maximum suppression technique'''
  img = Image.open(fname)

  predictions = non_maximum_suppresion(np.asarray(img), boosters)

  fig, ax = plt.subplots(1)
  ax.imshow(img)
  ax.axis('off')

  for row in range(predictions.shape[0]):
    for col in range(predictions.shape[1]):
      if predictions[row, col] >= 0:
        ax.add_patch(draw_rect((col * 64 + 32, row * 64 + 32)))

  if outfname:
    plt.savefig(outfname, dpi=1200)
  plt.show()

def eval_image(int_test_img_rep, boosters):
  '''Return predicted face indices for N * 64 * 64 int_img_rep'''
  faces_idx = np.arange(int_test_img_rep.shape[0])

  for i in range(boosters.shape[0]):
    booster, threshold = boosters[i]
    hypotheses = eval_booster(int_test_img_rep, booster, threshold)
    int_test_img_rep = int_test_img_rep[hypotheses >= 0]
    faces_idx = faces_idx[hypotheses >= 0]

    print('Remaining Test Points:', faces_idx.shape[0])

  return faces_idx

def label_image_naive(fname, boosters, int_test_img_rep=None, outfname=None):
  '''Label the image by sliding a 64 * 64 window'''
  img = Image.open(fname)

  patches = view_as_windows(np.asarray(img), window_shape=(64, 64), step=32)
  if int_test_img_rep is None:
    int_test_img_rep = compute_integral_image(patches.reshape(
      (patches.shape[0] * patches.shape[1], 64, 64)))

  faces_idx =  eval_image(int_test_img_rep, boosters)
  faces_coord = [(idx % 49 * 32, idx // 49 * 32) for idx in faces_idx]

  fig, ax = plt.subplots(1)
  ax.imshow(img)
  ax.axis('off')

  for i in range(len(faces_coord)):
    ax.add_patch(draw_rect(faces_coord[i]))

  if outfname:
    plt.savefig(outfname, dpi=1200)
  plt.show()