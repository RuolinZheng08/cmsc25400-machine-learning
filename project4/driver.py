#!/usr/bin/env python3

from viola_jones import *
from postprocessing import *

def main():
  int_test_img_rep, boosters = None, None
  if os.path.exists('cached/boosters.npy'):
    boosters = np.load('cached/boosters.npy')
  else:
    int_img_rep, feat_lst, labels = preprocess_data()
    boosters = cascade(int_img_rep, feat_lst, labels)
    np.save('cached/boosters.npy', boosters)

  img = Image.open('test_img.jpg')
  if os.path.exists('cached/int_test_img_rep.npy'):
    int_test_img_rep = np.load('cached/int_test_img_rep.npy')
  else:
    patches = extract_patches(np.asarray(img))
    int_test_img_rep = compute_integral_image(patches)

  faces_idx = eval_image(int_test_img_rep, boosters)

  label_image(img, faces_idx)

if __name__ == '__main__':
  main()