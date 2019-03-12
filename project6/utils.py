import numpy as np
from torch.nn.utils.rnn import pad_sequence

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

def batch_to_tensor(embedding, batch):
  '''Return tensor of shape seq_len * batch_size * 200'''
  ret = []
  for line in batch:
    ret.append(line_to_tensor(embedding, line))
  return pad_sequence(ret)

def compute_accuracy(pred, truth):
  '''Compares the values from two numpy arrays'''
  diff = (pred != truth).any(axis=-1)
  return 1 - np.count_nonzero(diff) / truth.shape[1] / truth.shape[0]

def closest_vecs(embed_vecs, vecs):
  ''''Given two numpy arrays, return the closest vectors in a numpy array'''
  res = np.linalg.norm(vecs - embed_vecs, axis=-1)
  idx = np.argmin(res, axis=1)
  return idx, embed_vecs[idx].squeeze()

def vecs_to_line(embed_key_vecs, idx):
  '''Given a numpy array of string keys and indices, returns a string'''
  return ' '.join(list(embed_key_vecs[idx].squeeze()))