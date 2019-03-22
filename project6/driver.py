#!/usr/bin/env python3

import matplotlib.pyplot as plt
import csv
from lstm import *
from utils import *

def main():
  embed = torch.load('embed')
  embed_vecs = torch.stack(list(embed.values())).numpy()[:, None]
  embed_key_vecs = np.array(list(embed.keys()))
  embed_tup = embed, embed_vecs, embed_key_vecs
  glove = torch.load('glove')
  glove_vecs = torch.stack(list(glove.values())).numpy()[:, None]
  glove_key_vecs = np.array(list(glove.keys()))
  glove_tup = glove, glove_vecs, glove_key_vecs

  train_first, train_second = line_pairs('bobsue.seq2seq.train.tsv')
  dev_first, dev_second = line_pairs('bobsue.seq2seq.dev.tsv')
  test_first, test_second = line_pairs('bobsue.seq2seq.test.tsv')

  model = LSTM(200, 200)
  opt = optim.SGD(model.parameters(), lr=1, weight_decay=0.001)
  scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
  t_losses, t_accus, d_losses, d_accus = [], [], [], []
  verbose = False

  for e in range(10):
    print('Epoch ', e)
    scheduler.step()
    t_loss, t_accu = train(model, glove_tup, train_first, train_second,
      verbose=verbose)
    d_loss, d_accu = validate(model, glove_tup, dev_first, dev_second,
      verbose=verbose)
    if verbose:
      print('e {} | loss {}, accu {}'.format(e, 
        np.mean(t_loss), np.mean(t_accu)))
      print('v    | ', np.mean(d_loss), np.mean(d_accu))      

    t_losses.extend(t_loss)
    t_accus.extend(t_accu)
    d_losses.extend(d_loss)
    d_accus.extend(d_accu)

  print('Done Training. Writing to File.')
  preds = predict(model, glove_tup, test_first)
  with open('bobsue.seq2seq.pred.tsv ', 'wt') as fout:
    tsv_writer = csv.writer(fout, delimiter='\t')
    for first, second in zip(test_first, preds):
      tsv_writer.writerow([first, second])

  # for i, elm in enumerate([t_losses, t_accus, d_losses, d_accus]):
  #   plt.plot(elm)
  #   # plt.savefig(str(i))
  #   plt.show()
  #   plt.clf()

if __name__ == '__main__':
  main()