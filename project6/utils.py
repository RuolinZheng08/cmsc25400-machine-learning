import torch
import numpy as np
import re

def rand_embedding(words):
  '''Generate random word embeddings'''
  embed = {}
  for word in words:
    embed[word] = torch.FloatTensor(2, 3).uniform_(-0.1, 0.1)
  return embed

def line_pairs(fname):
  '''Return two lists of strings given a file'''
  with open(fname, 'rt') as f:
    lines = re.split(r'\n', f.read().rstrip().lower())
  lines = [re.split(r'\t', line) for line in lines]
  first, second = zip(*lines)
  return first, second

def line_to_tensor(embedding, line):
  '''Return tensor of shape seq_len * 1 * 200'''
  words = re.split(r'\s', line)
  ret = torch.zeros(len(words), 200)
  for idx, word in enumerate(words):
    ret[idx] = embedding[word]
  return ret.unsqueeze(1)

def compute_accuracy(pred, truth):
  '''Compare the values from two numpy arrays'''
  diff = (pred != truth).any(axis=-1)
  return 1 - np.count_nonzero(diff) / truth.shape[1] / truth.shape[0]

def closest_vecs(embed_vecs, vecs):
  ''''Given two numpy arrays, return the closest vectors in a numpy array'''
  res = np.linalg.norm(vecs - embed_vecs, axis=-1)
  idx = np.argmin(res, axis=1)
  return idx, embed_vecs[idx].squeeze()

def vecs_to_line(embed_key_vecs, idx):
  '''Given a numpy array of string keys and indices, return a string'''
  if idx.size == 1:
    return embed_key_vecs[idx.item()]
  else:
    return ' '.join(list(embed_key_vecs[idx].squeeze()))

def string_accuracy(pred, truth):
  '''Compare the words of two strings'''
  pred = re.split(r'\s', pred)
  truth = re.split(r'\s', truth)
  lp, lt = len(pred), len(truth)
  if lp < lt:
    pred.extend([' '] * (lt - lp))
  count = 0
  for i in range(lt):
    if pred[i] != truth[i]:
      count += 1
  return (lt - count) / lt