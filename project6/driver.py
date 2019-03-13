from utils import *
from lstm import *

def main():
  embed = torch.load('embed')
  embed_vecs = torch.stack(list(embed.values())).numpy()[:, None]
  embed_key_vecs = np.array(list(embed.keys()))
  glove = torch.load('glove')
  glove_vecs = torch.stack(list(glove.values())).numpy()[:, None]
  glove_key_vecs = np.array(list(glove.keys()))

  train_first, train_second = line_pairs('bobsue.seq2seq.train.tsv')
  dev_first, dev_second = line_pairs('bobsue.seq2seq.dev.tsv')
  test_first, test_second = line_pairs('bobsue.seq2seq.test.tsv')

  model = LSTM(200, 200)
  train()