import numpy as np
from torch.nn.utils.rnn import pad_sequence

def rand_embedding(words):
  embed = {}
  for word in words:
    embed[word] = np.random.uniform(-0.1, 0.1, (200,))
  return embed

def line_pairs(fname):
  with open(fname, 'rt') as f:
    lines = re.split(r'\n', f.read().rstrip())
  lines = [re.split(r'\t', line) for line in lines]
  first, second = zip(*lines)
  return first, second

def line_to_tensor(embedding, line):
  '''Return tensor of shape seq_len * 200'''
  words = re.split(r'\s', line)
  ret = torch.zeros(len(words), 200)
  for idx, word in enumerate(words):
    ret[idx] = torch.from_numpy(embedding[word])
  return ret

def batch_to_tensor(embedding, batch):
  '''Return tensor of shape seq_len * batch_size * 200'''
  ret = []
  for line in batch:
    ret.append(line_to_tensor(embedding, line))
  return pad_sequence(ret)