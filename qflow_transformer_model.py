import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

class QTransformer(nn.Module):
    def __init__(self, s_dim, a_dim, a_bins, alpha, num_layers=4, nhead=2, action_min=-1.0, action_max=1.0, hdim=1024):
        super(QTransformer, self).__init__()
        self.hdim=hdim
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bins = a_bins
        self.alpha = alpha
        self.num_layers = num_layers
        self.nhead = nhead
        self.action_min = action_min
        self.action_max = action_max
        
        self.s_embed = nn.Linear(s_dim, hdim)
        self.a_embed = nn.Linear(1, hdim)
        self.positional_encoding = nn.Embedding(a_dim, hdim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead, dim_feedforward=hdim, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.q_head = nn.Linear(hdim, a_bins)
        
    def quantize_action(self, a):
        rescaled_action = (a - self.action_min) / (self.action_max - self.action_min) * (self.a_bins - 1)
        discrete_action = torch.round(rescaled_action).long()
        return discrete_action
    
    def dequantize_action(self, a):
        bin_width = (self.action_max - self.action_min) / self.a_bins
        continuous_action = self.action_min + (a + 0.5) * bin_width
        return continuous_action
        
    def forward(self, s, a=None, is_causal=True):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      x = self.s_embed(s)
      #print("shape of state embed x:", x.shape)

      seq_len = 1
      if a is not None:
          a_emb = self.a_embed(a)
          #print("shape of action embed (a_emb):", a_emb.shape)
          pos_enc_size = (a.shape[1], self.hdim)
          positional_encoding = nn.Embedding(*pos_enc_size)
          a_emb += positional_encoding.weight[:a.shape[1], :].to(device)

          if len(a_emb.shape) < len(x.shape):
            #match dim of x
            a_emb = a_emb.unsqueeze(1)
          x = torch.cat([x, a_emb], dim=1)
          
          seq_len = x.shape[1]
      else:
          seq_len = x.shape[0]
          
      if len(x.shape) == 3:
          seq_len = x.shape[1]
      #print("x shape after concatenation with action embed:", x.shape)

      x = self.transformer(x, mask=torch.nn.Transformer.generate_square_subsequent_mask(seq_len))
      Q = self.q_head(x)
      # Reduce dim to [128, 1]
      if Q.shape[1] > 1:
          Q = torch.mean(Q, dim=1, keepdim=True)

      V = torch.logsumexp(Q / self.alpha, dim=-1) * self.alpha
      pi = F.softmax(Q / self.alpha, dim=-1)
      # Ensure pi to be valid prob distribution
      pi = pi.clamp_min(1e-20) 
      logp = F.log_softmax(Q / self.alpha, dim=-1)

      return Q, V, pi, logp

    def sample_action(self, s, return_entropy=False, exploration_alpha=0.1):
        alpha_temp = self.alpha
        self.alpha = exploration_alpha
        s = s.unsqueeze(1)
        a = None
        for i in range(self.a_dim):
            _, _, pi,_ = self.forward(s, a)
            dist = torch.distributions.Categorical(pi, validate_args=False)
            a_i = dist.sample().unsqueeze(2)
            a_i = self.dequantize_action(a_i)
            if a is None:
                a = a_i
            else:
                a = torch.cat([a, a_i[:, -1:]], dim=1)
        self.alpha = alpha_temp
        if return_entropy:
            entropy = -dist.log_prob(self.quantize_action(a_i).squeeze(2))
            return a.squeeze(2), entropy.detach().sum(1).mean()
        return a.squeeze(2)

class GFN(nn.Module):
    def __init__(self, s_dim, a_dim, a_bins, alpha=1.0, action_min=-1.0, action_max=1.0, gfn_batch_size=64, gfn_lr=1e-3, num_layers=4, nhead=2, hdim=1024):
        super(GFN, self).__init__()
        self.hdim = hdim
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bins = a_bins
        self.alpha = alpha
        self.action_min = action_min
        self.action_max = action_max
        self.gfn_batch_size = gfn_batch_size
        
        self.q_transformer = QTransformer(s_dim, a_dim, a_bins, alpha, num_layers=num_layers, nhead=nhead, action_min=action_min, action_max=action_max, hdim=hdim)
        
        self.q1 = QTransformer(s_dim, a_dim, a_bins, alpha, num_layers=num_layers, nhead=nhead, action_min=action_min, action_max=action_max, hdim=hdim)
        self.q2 = QTransformer(s_dim, a_dim, a_bins, alpha, num_layers=num_layers, nhead=nhead, action_min=action_min, action_max=action_max, hdim=hdim)
        self.q1_target = QTransformer(s_dim, a_dim, a_bins, alpha, num_layers=num_layers, nhead=nhead, action_min=action_min, action_max=action_max, hdim=hdim)
        self.q2_target = QTransformer(s_dim, a_dim, a_bins, alpha, num_layers=num_layers, nhead=nhead, action_min=action_min, action_max=action_max, hdim=hdim)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.init_opt(lr=gfn_lr)
        
    def init_opt(self, lr=1e-4):
        self.opt = torch.optim.Adam(self.q_transformer.parameters(), lr=lr)

    def quantize_action(self, a):
        rescaled_action = (a - self.action_min) / (self.action_max - self.action_min) * (self.a_bins - 1)
        discrete_action = torch.round(rescaled_action).long()
        return discrete_action
    
    def dequantize_action(self, a):
        bin_width = (self.action_max - self.action_min) / self.a_bins
        continuous_action = self.action_min + (a + 0.5) * bin_width
        return continuous_action
    
    def train_GFN(self, s):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s = s.repeat_interleave(self.gfn_batch_size, 0)
        bs = s.shape[0]
        logpf = torch.zeros((bs,), device=device)
        self.opt.zero_grad()
        print("Checkpoint 1: Initialization completed")

        total_log_prob = torch.zeros(bs, device=device)
        print("Checkpoint 2: Initialization of total_log_prob completed")
        Q, V, pi, logp = self.q_transformer.forward(s)
        dist = torch.distributions.Categorical(pi, validate_args=False)
        a_i = dist.sample().unsqueeze(1)
        logpf += logp[torch.arange(bs), a_i.squeeze(1)]
        print("Checkpoint 3: Sampling completed")
        a_logp = logp.gather(1, a_i).squeeze(1)
        total_log_prob += a_logp

        a_i = self.dequantize_action(a_i)
        a = a_i
        for i in range(1, self.a_dim):
            Q, V, pi, logp = self.q_transformer.forward(s)
            dist = torch.distributions.Categorical(pi, validate_args=False)
            a_i = dist.sample().unsqueeze(1)
            logpf += logp[torch.arange(bs), a_i.squeeze(1)]
            a_logp = logp.gather(1, a_i).squeeze(1)
            total_log_prob += a_logp
            a_i = self.dequantize_action(a_i)
            a = torch.cat([a, a_i], dim=1)

        logreward_1, _, _,_ = self.q1_target(s)
        logreward_2, _, _,_ = self.q2_target(s)
        logreward = torch.min(logreward_1, logreward_2).flatten()

        delta_x = torch.tensor((self.action_max - self.action_min) / self.a_bins, device=device)
        continuous_entropy_adj = total_log_prob - torch.log(delta_x) * self.a_dim

        neg_log_pf = self.alpha * (-logpf)
        logZ = (neg_log_pf + logreward).detach().view(-1, self.gfn_batch_size).mean(1).repeat_interleave(self.gfn_batch_size, 0)
        loss = 0.5 * ((-neg_log_pf + logZ - logreward) ** 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_transformer.parameters(), max_norm=1.0)
        self.opt.step()
        print("Checkpoint 4: Training step completed")

        return loss.detach().cpu().numpy(), logZ.mean().detach().cpu().numpy(), continuous_entropy_adj.mean().item()


    def forward(self, s):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      bs = s.shape[0]
      logpf = torch.zeros((bs,), device=device)
      
      Q, V, pi, logp = self.q_transformer.forward(s)
      dist = torch.distributions.Categorical(pi, validate_args=False)
      a_i = dist.sample().unsqueeze(1)
      logpf += logp[torch.arange(bs), a_i.squeeze(1)]
      a_i = self.dequantize_action(a_i)
      a = a_i
      for i in range(1, self.a_dim):
          Q, V, pi, logp = self.q_transformer.forward(s, a)
          dist = torch.distributions.Categorical(pi, validate_args=False)
          a_i = dist.sample().unsqueeze(1)
          logpf += logp[torch.arange(bs), a_i.squeeze(1)]
          a_i = self.dequantize_action(a_i)
          a = torch.cat([a, a_i], dim=1)
      
      return a, logpf
