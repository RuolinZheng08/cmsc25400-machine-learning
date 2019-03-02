#!/usr/bin/env python3

from viola_jones import *
from postprocessing import *

def main():
  int_img_rep, feat_lst, labels = preprocess_data()
  boosters = cascade(int_img_rep, feat_lst, labels)
  label_image_naive('test_img.jpg', boosters)

if __name__ == '__main__':
  main()