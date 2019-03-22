import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *

class LSTMCell(nn.Module):
  '''An LSTMCell computes hidden and context for one word, (batch_size, n_in)'''
  def __init__(self, n_in, n_hid):
    super(LSTMCell, self).__init__()
    self.w_if = nn.Linear(n_in, n_hid)
    self.w_if.weight.data.uniform_(-0.1, 0.1)
    self.w_if.bias.data.zero_()
    self.w_ii = nn.Linear(n_in, n_hid)
    self.w_ii.weight.data.uniform_(-0.1, 0.1)
    self.w_ii.bias.data.zero_()
    self.w_ig = nn.Linear(n_in, n_hid)
    self.w_ig.weight.data.uniform_(-0.1, 0.1)
    self.w_ig.bias.data.zero_()
    self.w_io = nn.Linear(n_in, n_hid)
    self.w_io.weight.data.uniform_(-0.1, 0.1)
    self.w_io.bias.data.zero_()
    self.w_hf = nn.Linear(n_hid, n_hid)
    self.w_hf.weight.data.uniform_(-0.1, 0.1)
    self.w_hf.bias.data.zero_()
    self.w_hi = nn.Linear(n_hid, n_hid)
    self.w_hi.weight.data.uniform_(-0.1, 0.1)
    self.w_hi.bias.data.zero_()
    self.w_hg = nn.Linear(n_hid, n_hid)
    self.w_hg.weight.data.uniform_(-0.1, 0.1)
    self.w_hg.bias.data.zero_()
    self.w_ho = nn.Linear(n_hid, n_hid)
    self.w_ho.weight.data.uniform_(-0.1, 0.1)
    self.w_ho.bias.data.zero_()

  def forward(self, x, tup):
    hid, ctx = tup
    f = torch.sigmoid(self.w_if(x) + self.w_hf(hid))
    i = torch.sigmoid(self.w_ii(x) + self.w_hi(hid))
    g = torch.tanh(self.w_ig(x) + self.w_hg(hid))
    o = torch.sigmoid(self.w_io(x) + self.w_ho(hid))
    ctx = f * ctx + i * g
    hid = o * torch.tanh(ctx)
    return hid, ctx

class LSTM(nn.Module):
  '''An LSTM Net is a RNN using an LSTMCell, (seq_len, batch_size, n_in)'''
  def __init__(self, n_in, n_hid):
    super(LSTM, self).__init__()
    self.n_hid = n_hid
    self.lstmcell = LSTMCell(n_in, n_hid)
    self.fc = nn.Linear(n_in, n_in)

  def forward(self, x, tup):
    hid, ctx = tup
    if hid is None:
      hid = torch.zeros(x.shape[1], self.n_hid)
      ctx = torch.zeros(x.shape[1], self.n_hid)
    output = torch.zeros(x.shape[0], x.shape[1], self.n_hid)
    for t in range(x.shape[0]):
      hid, ctx = self.lstmcell(x[t], (hid, ctx))
      output[t] = hid
    return self.fc(output), (hid, ctx)

def train(model, opt, embed_tup, train_first, train_second, verbose=False):
  '''Train on test set'''
  embed, embed_vecs, embed_key_vecs = embed_tup
  criterion = nn.MSELoss()
  loss, accu = [], []

  for i in range(len(train_first)):
    opt.zero_grad()
    xs = line_to_tensor(embed, train_first[i])
    ys = line_to_tensor(embed, train_second[i])
    hid, ctx = None, None

    output, (hid, ctx) = model(xs, (hid, ctx))
    output, (hid, ctx) = model(ys, (hid, ctx))
    output, ys = output[:-1], ys[1:]

    idx, vecs = closest_vecs(embed_vecs, output.data.numpy()[:, None])
    
    loss_tens = criterion(output, ys)
    loss.append(loss_tens.item())
    accu.append(compute_accuracy(vecs[:, None], ys.numpy()))

    loss_tens.backward(retain_graph=True)
    opt.step()
  if verbose:
    print('e | ' + vecs_to_line(embed_key_vecs, idx))
  return loss, accu

def validate(model, embed_tup, dev_first, dev_second, verbose=False):
  '''Validate on dev set'''
  embed, embed_vecs, embed_key_vecs = embed_tup
  criterion = nn.MSELoss()
  loss, accu = [], []

  for i in range(len(dev_first)):
    xs = line_to_tensor(embed, dev_first[i])
    ys = line_to_tensor(embed, dev_second[i])
    hid, ctx = None, None

    output, (hid, ctx) = model(xs, (hid, ctx))
    output, (hid, ctx) = model(ys, (hid, ctx))
    output, ys = output[:-1], ys[1:]

    idx, vecs = closest_vecs(embed_vecs, output.data.numpy()[:, None])
    
    loss.append(criterion(output, ys).item())
    accu.append(compute_accuracy(vecs[:, None], ys.numpy()))
  if verbose:
    print('v | ' + vecs_to_line(embed_key_vecs, idx))
  return loss, accu

def predict(model, embed_tup, test_first):
  '''Predict on test set'''
  embed, embed_vecs, embed_key_vecs = embed_tup
  pred_lines = []
  for i in range(len(test_first)):
    line = []
    xs = line_to_tensor(embed, test_first[i])
    hid, ctx = None, None

    output, (hid, ctx) = model(xs, (hid, ctx))
    output, (hid, ctx) = model(line_to_tensor(embed, '<s>'), (hid, ctx))
    line.append('<s>')

    for j in range(21):
      idx, vecs = closest_vecs(embed_vecs, output.data.numpy()[:, None])
      word = vecs_to_line(embed_key_vecs, idx)
      line.append(word)
      if word == '</s>':
        break
      output, (hid, ctx) = model(torch.from_numpy(vecs[None, None, :]),
        (hid, ctx))

    pred_lines.append(' '.join(line))
  return pred_lines