import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
    
    
class QMLP(nn.Module):
    def __init__(self, state_dim, a_dim):
        super(QMLP, self).__init__()
        self.state_dim = state_dim
        h_dim = 256
        self.fc1 = nn.Linear(state_dim+a_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.gelu(self.ln1(self.fc1(x)))
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class ARMLP(nn.Module):
    def __init__(self, state_dim, a_bins):
        super(ARMLP, self).__init__()
        self.state_dim = state_dim
        h_dim = 256
        self.fc1 = nn.Linear(state_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, a_bins)
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, s, tau=1.0):
        x = F.gelu(self.ln1(self.fc1(s)))
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.fc3(x)/tau
        pi = self.softmax(x)
        logp = self.log_softmax(x)
        return pi, logp
    
# AutoRegressive Q function
class GFN(nn.Module):
    def __init__(self, s_dim, a_dim, a_bins, alpha=1.0, action_min=-1.0, action_max=1.0, gfn_batch_size=64, gfn_lr=1e-3):
        super(GFN, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bins = a_bins
        self.alpha = alpha
        self.action_min = action_min
        self.action_max = action_max
        self.gfn_batch_size = gfn_batch_size
        
        self.mlp = nn.ModuleList([ARMLP(s_dim+i,a_bins) for i in range(a_dim)])
        self.q1 = QMLP(s_dim, a_dim)
        self.q2 = QMLP(s_dim, a_dim)
        self.q1_target = QMLP(s_dim, a_dim)
        self.q2_target = QMLP(s_dim, a_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        #self.logpb = -torch.tensor(a_dim*np.log(a_bins)).cuda()
        self.init_opt(lr=gfn_lr)
        
    def init_opt(self, lr=1e-4):
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        
    def quantize_action(self, a):
        rescaled_action = (a - self.action_min) / (self.action_max - self.action_min) * (self.a_bins - 1)
        discrete_action = torch.round(rescaled_action).long()
        return discrete_action
    
    def dequantize_action(self, a):
        #a = torch.argmax(a, dim=2, keepdim=True)
        bin_width = (self.action_max - self.action_min) / self.a_bins
        continuous_action = self.action_min + (a + 0.5) * bin_width
        return continuous_action
    
    def train_GFN(self, s):
        device = torch.device('cuda')
        s = s.repeat_interleave(self.gfn_batch_size, 0)
        bs = s.shape[0]
        logpf = torch.zeros((bs,), device=device)
        self.opt.zero_grad()
        
        pi, logp = self.forward_once(s)
        dist = torch.distributions.Categorical(pi)
        a_i = dist.sample().unsqueeze(1)
        logpf += logp[torch.arange(bs), a_i.squeeze(1)]
        a_i = self.dequantize_action(a_i)
        a = a_i
        for i in range(1, self.a_dim):
            pi, logp = self.forward_once(s, a)
            dist = torch.distributions.Categorical(pi)
            a_i = dist.sample().unsqueeze(1)
            logpf += logp[torch.arange(bs), a_i.squeeze(1)]
            a_i = self.dequantize_action(a_i)
            a = torch.cat([a, a_i], dim=1)
        logreward_1 = self.q1(s, a)
        logreward_2 = self.q2(s, a)
        logreward = torch.min(logreward_1, logreward_2).flatten()
        neg_log_pf = self.alpha*(-logpf)
        logZ = (neg_log_pf + logreward).detach().view(-1, self.gfn_batch_size).mean(1).repeat_interleave(self.gfn_batch_size, 0)
        loss = 0.5*((-neg_log_pf + logZ - logreward)**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), max_norm=1.0)
        self.opt.step()
        return loss.detach().cpu().numpy(), logZ.mean().detach().cpu().numpy()
        
    def forward_once(self, s, a=None, tau=None):
        if tau is None:
            tau = 1.0
        mlp_idx = 0
        if a is not None:
            s = torch.cat([s, a], dim=1)
            mlp_idx = a.shape[1]
        pi, logp = self.mlp[mlp_idx](s, tau=tau)
        return pi, logp
    
    def forward(self, s):
        device = torch.device('cuda')
        bs = s.shape[0]
        logpf = torch.zeros((bs,), device=device)
        
        pi, logp = self.forward_once(s)
        dist = torch.distributions.Categorical(pi)
        a_i = dist.sample().unsqueeze(1)
        logpf += logp[torch.arange(bs), a_i.squeeze(1)]
        a_i = self.dequantize_action(a_i)
        a = a_i
        for i in range(1, self.a_dim):
            pi, logp = self.forward_once(s, a)
            dist = torch.distributions.Categorical(pi)
            a_i = dist.sample().unsqueeze(1)
            logpf += logp[torch.arange(bs), a_i.squeeze(1)]
            a_i = self.dequantize_action(a_i)
            a = torch.cat([a, a_i], dim=1)
        return a, logpf