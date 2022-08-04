import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from nlp.decoder import LSTMDecoder
from nlp.lstm import LSTMEncoder
from lossUtils import Loss
from nlp.attention import attention

import pdb

import pickle as pkl
import numpy as np

class TeacherForcing():
  '''
  Sends True at the start of training, i.e. Use teacher forcing maybe.
  Progressively becomes False by the end of training, start using gt to train
  '''
  def __init__(self, max_epoch):
    self.max_epoch = max_epoch

  def __call__(self, epoch, batch_size=1):
    p = epoch*1./self.max_epoch
    random = torch.rand(batch_size)
    return (p < random).double()  

## Sequence to Sequence AutoEncoder

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=1):
    super(Encoder, self).__init__()
    self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)

  def forward(self, x):
    outputs, h_n = self.rnn(x)
    ## TODO include attention 
    return outputs
  
class TrajectoryPredictor(nn.Module):
  def __init__(self, pose_size, trajectory_size, hidden_size):
    super(TrajectoryPredictor, self).__init__()
    self.lp = nn.Linear(hidden_size, pose_size)
    self.fc = nn.Linear(pose_size+hidden_size, trajectory_size)


  def forward(self, x):
    pose_vector = self.lp(x)
    trajectory_vector = self.fc(torch.cat((pose_vector, x), dim=-1))
    mixed_vector = torch.cat((trajectory_vector, pose_vector), dim=-1)
    return mixed_vector

class DecoderCell(nn.Module):
  def __init__(self, hidden_size, pose_size, trajectory_size, use_h=False, use_tp=True, use_lang=False):
    super(DecoderCell, self).__init__()
    self.use_h = 1 if use_h else 0
    self.use_lang = 1 if use_lang else 0
    self.rnn = nn.GRUCell(input_size=pose_size+trajectory_size+hidden_size*(self.use_h+self.use_lang),
                          hidden_size=hidden_size)
    if use_tp:
      self.tp = TrajectoryPredictor(pose_size=pose_size,
                                    trajectory_size=trajectory_size,
                                    hidden_size=hidden_size)
    else:
      self.tp = nn.Linear(hidden_size, pose_size + trajectory_size)

    if self.use_lang:
      self.lin = nn.Linear(hidden_size+pose_size+trajectory_size, pose_size+trajectory_size)

  def forward(self, x, h):
    if self.use_h:
      x_ = torch.cat([x,h], dim=-1)
    else:
      x_ = x
    h_n = self.rnn(x_, h)
    ## TODO add attention
    tp_n = self.tp(h_n)
    if self.use_lang:
      y = self.lin(x) + tp_n
    else:
      y = x + tp_n
    return y, h_n

class Decoder(nn.Module):
  def __init__(self, hidden_size, pose_size, trajectory_size,
               use_h=False, start_zero=False, use_tp=True,
               use_lang=False, use_attn=False):
    super(Decoder, self).__init__()
    self.input_size = pose_size + trajectory_size
    self.cell = DecoderCell(hidden_size, pose_size, trajectory_size,
                            use_h=use_h, use_tp=use_tp, use_lang=use_lang)
    ## Hardcoded to reach 0% Teacher forcing in 10 epochs
    self.tf = TeacherForcing(0.1)
    self.start_zero = start_zero
    self.use_lang = use_lang
    self.use_attn = use_attn
    
  def forward(self, h, time_steps, gt, epoch=np.inf, attn=None):
    if self.use_lang:
      lang_z = h
    if self.start_zero:
      x = h.new_zeros(h.shape[0], self.input_size)
      x = h.new_tensor(torch.rand(h.shape[0], self.input_size))
    else:
      x = gt[:, 0, :] ## starting point for the decoding 

    Y = []
    for t in range(time_steps):
      if self.use_lang:
        if self.use_attn:  ### calculate attention at each time-step
          lang_z = attn(h)          
        x, h = self.cell(torch.cat([x, lang_z], dim=-1), h)
      else:
        x, h = self.cell(x, h)
      Y.append(x.unsqueeze(1))
      if t > 0:
        mask = self.tf(epoch, h.shape[0]).double().view(-1, 1).to(x.device)
        x = mask * gt[:, t-1, :] + (1-mask) * x
    return torch.cat(Y, dim=1)

  def sample(self, h, time_steps, start, attn=None):
    if self.use_lang:
      lang_z = h

    #x = torch.rand(h.shape[0], self.input_size).to(h.device).to(h.dtype)
    x = start ## starting point for the decoding 
    Y = []
    for t in range(time_steps):
      if self.use_lang:
        if self.use_attn:
          lang_z = attn(h)
        x, h = self.cell(torch.cat([x, lang_z], dim=-1), h)
      else:
        x, h = self.cell(x, h)
      Y.append(x.unsqueeze(1))
    return torch.cat(Y, dim=1)
    
class Seq2Seq(nn.Module):
  ## TODO add an integrator and differentiator to handle delta values end2end
  def __init__(self, hidden_size, pose_size, trajectory_size,
               use_h=False, start_zero=False, use_tp=True,
               use_lang=False, use_attn=False, **kwargs):
    super(Seq2Seq, self).__init__()
    if use_attn: ## use_lang must be true if use_attn is true
      use_lang=True
    ## TODO take root rotation out of Trajectory Predictor
    #pose_size += 4
    #trajectory_size -= 4
    input_size = pose_size + trajectory_size
    self.enc = Encoder(input_size, hidden_size)
    self.dec = Decoder(hidden_size, pose_size, trajectory_size,
                       use_h=use_h, start_zero=start_zero,
                       use_tp=use_tp, use_lang=use_lang,
                       use_attn=use_attn)

  def forward(self, x, train=True, epoch=np.inf, attn=None):
    time_steps = x.shape[1]
    enc_vector = self.enc(x)[:, -1, :]
    dec_vector = self.dec(enc_vector, time_steps, gt=x, epoch=epoch, attn=attn)
    return dec_vector, []

class Seq2SeqConditioned9(nn.Module):
  ''' 
  Sentence conditioned pose generation
  if train:
    choose from l2p and p2p
  else:
    l2p
  Seq2SeqKwargs = {hidden_size, 
                   use_h:False,
                   use_lang:False, 
                   use_tp:True,
                   start_zero:False, 
                   s2v:'lstm' or 'bert'}))

  *JL2P*
  '''
  def __init__(self, chunks, input_size=300, Seq2SeqKwargs={}, load=None):
    super(Seq2SeqConditioned9, self).__init__()
    self.chunks = chunks
    self.hidden_size = Seq2SeqKwargs['hidden_size']
    self.trajectory_size = Seq2SeqKwargs['trajectory_size']
    self.pose_size = Seq2SeqKwargs['pose_size']
    self.seq2seq = Seq2Seq(**Seq2SeqKwargs)
    if load:
      self.seq2seq.load_state_dict(pkl.load(open(load, 'rb')))
      print('Seq2Seq Model Loaded')
    else:
      print('Seq2Seq Model not found. Initialising randomly')


    self.sentence_enc = LSTMEncoder(self.hidden_size)
    
  def js_divergence(self, p, q):
    pdb.set_trace()
    m = torch.log((p+q)/2)
    return F.kl_div(m, p, reduce='sum') + F.kl_div(m, q, reduce='sum')


  def forward(self, pose, s2v, e2v, train=False, epoch=np.inf):
    pose_enc = self.seq2seq.enc(pose)
    print('s2v: ', s2v)
    print('e2v: ', e2v)
    language_z, _ = self.sentence_enc(s2v)
    print(language_z.shape)
    print()
    time_steps = pose.shape[-2]
    
    if torch.rand(1).item() > 0.5 or not train:
      pose_dec = self.seq2seq.dec(language_z, time_steps, gt=pose)
    else:
      pose_dec = self.seq2seq.dec(pose_enc[:, -1, :], time_steps, gt=pose)
    #internal_losses = [F.mse_loss(pose_enc[:, -1, :], language_z, reduction='mean')]
    internal_losses = []
    return pose_dec, internal_losses

  def sample(self, s2v, time_steps, start):
    language_z, _ = self.sentence_enc(s2v)
    pose_dec = self.seq2seq.dec.sample(language_z, time_steps, start)
    #internal_losses = [F.mse_loss(pose_enc[:, -1, :], language_z, reduction='mean')]
    internal_losses = []
    return pose_dec, internal_losses  
