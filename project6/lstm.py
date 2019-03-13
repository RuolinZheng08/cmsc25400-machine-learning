import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

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

def validate(model, embed_tup, dev_first, dev_second):
  '''Validate on dev set'''
  embed, embed_vecs, embed_key_vecs = embed_tup
  criterion = nn.MSELoss()
  loss, accuracy = 0, 0
  n_dev = len(dev_first)
  for i in range(n_dev):
    xs = line_to_tensor(embed, dev_first[i])
    ys = line_to_tensor(embed, dev_second[i])
    hid, ctx = None, None

    output, (hid, ctx) = model(xs, (hid, ctx))
    output, (hid, ctx) = model(ys, (hid, ctx))
    idx, vecs = closest_vecs(glove_vecs, output.data.numpy()[:, None])
    
    loss += criterion(output, ys).item()
    accuracy += compute_accuracy(vecs[:, None], ys.numpy())

  return loss / n_dev, accuracy / n_dev

def train(model, embed_tup, train_first, train_second, dev_first, dev_second,
  n_epochs=30):
  '''Train on test set'''
  embed, embed_vecs, embed_key_vecs = embed_tup
  opt = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)
  criterion = nn.MSELoss()
  n_train = len(train_first)
  t_losses, t_accus, d_losses, d_accus = [], [], [], []

  for e in range(n_epochs):
    loss, accuracy = 0, 0
    for i in range(n_train):
      xs = line_to_tensor(embed, train_first[i])
      ys = line_to_tensor(embed, train_second[i])
      hid, ctx = torch.zeros(1, 1, 200), torch.zeros(1, 1, 200)

      output, (hid, ctx) = model(xs, (hid, ctx))
      output, (hid, ctx) = model(ys, (hid, ctx))
      idx, vecs = closest_vecs(glove_vecs, output.data.numpy()[:, None])
      
      loss_tens = criterion(output, ys)
      loss += loss_tens.item()
      accuracy += compute_accuracy(vecs[:, None], ys.numpy())

      loss_tens.backward(retain_graph=True)
      opt.step()

    loss /= n_train
    accuracy /= n_train
    print('e {} | loss {}, accu {}'.format(e, loss, accuracy))
    print(vecs_to_line(embed_key_vecs, idx))
    if e % 10 == 0:
      d_loss, d_accu = validate(model, embed_tup, dev_first, dev_second)
      print('v | ', d_loss, d_accu)

    t_losses.append(loss)
    t_accus.append(accuracy)
    d_losses.append(d_losses)
    d_accus.append(d_accu)

  return t_losses, t_accus, d_losses, d_accus

def predict(model, embed_tup, test_first):
  embed, embed_vecs, embed_key_vecs = embed_tup
  pred_lines = []
  for i in range(len(test_first)):
    line = ''
    xs = line_to_tensor(embed, test_first[i])
    hid, ctx = None, None

    output, (hid, ctx) = model(xs, (hid, ctx))
    for j in range(21):
      output, (hid, ctx) = model(hid, (hid, ctx))
      idx, vecs = closest_vecs(embed_vecs, hid.data.numpy()[:, None])
      word = vecs_to_line(embed_key_vecs, idx)
      line += word + ' '
    pred_lines.append(line)
  return pred_lines