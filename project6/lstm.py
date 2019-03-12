import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import re

class LSTMCell(nn.Module):
  '''An LSTMCell computes hidden and context for one word, (batch_size, n_in)'''
  def __init__(self, n_in, n_hid):
    super(LSTMCell, self).__init__()
    self.w_if = nn.Linear(n_in, n_hid)
    self.w_ii = nn.Linear(n_in, n_hid)
    self.w_ig = nn.Linear(n_in, n_hid)
    self.w_io = nn.Linear(n_in, n_hid)
    self.w_hf = nn.Linear(n_hid, n_hid)
    self.w_hi = nn.Linear(n_hid, n_hid)
    self.w_hg = nn.Linear(n_hid, n_hid)
    self.w_ho = nn.Linear(n_hid, n_hid)

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

  def forward(self, x, tup):
    hid, ctx = tup
    if hid is None:
      hid = torch.zeros(x.shape[1], self.n_hid)
      ctx = torch.zeros(x.shape[1], self.n_hid)
    output = torch.zeros(x.shape[0], x.shape[1], self.n_hid)
    for t in range(x.shape[0]):
      hid, ctx = self.lstmcell(x[t], (hid, ctx))
      output[t] = hid
    return output, (hid, ctx)

def train(model, embedding, xs, ys):
  embed_vecs = torch.stack(list(embedding.values())).numpy()[:, None]
  y_vecs = ys.numpy()

  opt = optim.SGD(model.parameters(), lr=5)
  criterion = nn.MSELoss()
  
  hid, ctx = None, None
  for e in range(20):
    opt.zero_grad()

    output, (hid, ctx) = model(xs, (hid, ctx))
    output, (hid, ctx) = model(ys, (hid, ctx))

    loss = criterion(output, ys)
    
    vecs = closest_vecs(embed_vecs, output.data.numpy()[:, None])
    accuracy = compute_accuracy(vecs, y_vecs)
    loss.backward(retain_graph=True)
    opt.step()
    print('Epoch {} | loss: {},  accuracy: {}'.format(e, loss.item(), accuracy))
  
  return output, (hid, ctx)