import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PosEnc(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.register_buffer('freq_bands', (2.0 ** torch.arange(L).float()) * math.pi)
        self.register_buffer('freq_weights', torch.ones(L))

    @torch.no_grad()
    def set_window_by_iter(self, step, end_step):
        if not end_step or end_step <= 0:
            w = torch.ones(self.L, device=self.freq_weights.device)
        else:
            p = max(0.0, min(1.0, float(step) / float(end_step)))
            k = torch.arange(self.L, device=self.freq_weights.device, dtype=torch.float32)
            w = (p * self.L - k).clamp_(0.0, 1.0)
        self.freq_weights.copy_(w)

    def forward(self, x):
        f = self.freq_bands.to(x.device)               # [L]
        xb = x[..., None, :] * f[:, None]              # [..., L, C]
        s, c = torch.sin(xb), torch.cos(xb)
        w = self.freq_weights.to(x.device)[:, None]    # [L,1]
        s, c = s * w, c * w
        y = torch.cat([s, c], dim=-1)                  # [..., L, 2C]
        return y.view(*x.shape[:-1], -1)

class TinyNeRF_PE(nn.Module):
    def __init__(self, Lx=10, Ld=4, hidden=256):
        super().__init__()
        self.pe_xyz = PosEnc(Lx)
        self.pe_dir = PosEnc(Ld)
        in_xyz = 3 + 3 * 2 * Lx
        in_dir = 3 * 2 * Ld
        self.fc1 = nn.Linear(in_xyz, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden + in_xyz, hidden)
        self.fc_sigma = nn.Linear(hidden, 1)
        self.fc_feat  = nn.Linear(hidden, hidden)
        self.fc_dir   = nn.Linear(hidden + in_dir, hidden // 2)
        self.fc_rgb   = nn.Linear(hidden // 2, 3)

    def forward(self, pts, dirs=None):
        pe_xyz = self.pe_xyz(pts)
        x_in = torch.cat([pts, pe_xyz], -1)
        h = F.relu(self.fc1(x_in))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = torch.cat([h, x_in], -1)
        h = F.relu(self.fc4(h))
        sigma = F.softplus(self.fc_sigma(h))
        feat = F.relu(self.fc_feat(h))
        if dirs is not None:
            pe_d = self.pe_dir(dirs)
            h_col = torch.cat([feat, pe_d], -1)
        else:
            h_col = feat
        h_col = F.relu(self.fc_dir(h_col))
        rgb = torch.sigmoid(self.fc_rgb(h_col))
        return rgb, sigma
