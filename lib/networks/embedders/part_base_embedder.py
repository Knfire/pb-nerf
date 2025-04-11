from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sympy import nextprime
from lib.config import cfg
from termcolor import cprint



class Embedder(nn.Module):
    def __init__(self,
                 pid=-1,
                 partname='undefined',
                 bbox=np.array([
                     [0, 0, 0],
                     [1, 1, 1]
                 ]),
                 n_levels=16,
                 n_features_per_level=16,
                 b=1.38,
                 log2_hashmap_size=18,
                 base_resolution=2,
                 sum=True,
                 sum_over_features=True,
                 separate_dense=True,
                 use_batch_bounds=cfg.use_batch_bounds,
                 include_input=True,  # this will pass gradient better to input, but if you're using uvt, no need
                 device='cuda',  # Specify device
                 ):
        super().__init__()
        self.pid = pid
        self.partname = partname
        self.n_levels = n_levels
        self.include_input = include_input
        self.use_batch_bounds = use_batch_bounds
        self.device = device  # Set device
        self.n_entries_per_level = nextprime(2**log2_hashmap_size)

        self.b = b
        self.f = n_features_per_level
        self.base_resolution = base_resolution

        # Move bbox to device
        self.bounds = nn.Parameter(torch.tensor(np.array(bbox).reshape((2, 3))).float().to(self.device), requires_grad=False)

        # Calculate entries for each level and move to device
        self.entries_num = [int((self.base_resolution * self.b**i)) for i in range(self.n_levels)]
        self.entries_cnt = [self.entries_num[i] ** 3 for i in range(self.n_levels)]
        self.entries_size = [1 / (self.entries_num[i] - 1) for i in range(self.n_levels)]
        self.entries_min = [0 for i in range(self.n_levels)]

        self.entries_size = nn.Parameter(torch.tensor(self.entries_size).to(self.device), requires_grad=False)
        self.entries_num = nn.Parameter(torch.tensor(self.entries_num).to(self.device), requires_grad=False)
        self.entries_min = nn.Parameter(torch.tensor(self.entries_min).to(self.device), requires_grad=False)
        self.entries_cnt = nn.Parameter(torch.tensor(self.entries_cnt).to(self.device), requires_grad=False)
        self.entries_sum = nn.Parameter(self.entries_cnt.cumsum(dim=-1).to(self.device), requires_grad=False)

        self.start_hash = self.n_levels
        for i in range(n_levels):
            if self.entries_cnt[i] > self.n_entries_per_level:
                self.start_hash = i
                break
        self.len_hash = self.n_levels - self.start_hash
        self.separate_dense = separate_dense and self.start_hash  # when everything needs to be hashed

        if self.separate_dense:
            data = torch.zeros((self.n_levels, self.n_entries_per_level, self.f)).to(self.device)
            nn.init.kaiming_normal_(data)
            dense = torch.cat([data[i, :self.entries_cnt[i], :] for i in range(self.start_hash)], dim=0)
            hash = data[self.start_hash:, :, :]
            self.dense = nn.Parameter(dense)  # sum(non-hash), F
            self.hash = nn.Parameter(hash)  # H, T, F
        else:
            self.hash = nn.Parameter(torch.zeros((self.n_levels, self.n_entries_per_level, self.f)).to(self.device))  # H, T, F
            nn.init.kaiming_normal_(self.hash)

        self.offsets = nn.Parameter(torch.tensor([[0., 0., 0.],
                                                  [0., 0., 1.],
                                                  [0., 1., 0.],
                                                  [0., 1., 1.],
                                                  [1., 0., 0.],
                                                  [1., 0., 1.],
                                                  [1., 1., 0.],
                                                  [1., 1., 1.]]).float().to(self.device), requires_grad=False)

        self.sum = sum
        self.sum_over_features = sum_over_features
        self.out_dim = 0

        if self.sum:
            if self.sum_over_features:
                self.out_dim += self.n_levels
            else:
                self.out_dim += self.f
        else:
            self.out_dim += self.f * self.n_levels

        if include_input:
            self.out_dim += 3

    def forward(self, xyz: torch.Tensor, batch):
        if self.use_batch_bounds and 'iter_step' in batch and batch['iter_step'] == 1:
            self.bounds = nn.Parameter(batch['bounds'][0][self.pid].to(self.device), requires_grad=False)

        N, _ = xyz.shape  # N, 3
        xyz = (xyz.to(self.device) - self.bounds[0]) / (self.bounds[1] - self.bounds[0])  # normalized, N, 3

        ind_xyz = xyz[None].expand(self.n_levels, -1, -1)  # L, N, 3
        flt_xyz = ind_xyz / self.entries_size[:, None, None]  # L, N, 3
        int_xyz = (flt_xyz[:, :, None] + self.offsets[None, None]).long()  # L, N, 8, 3
        int_xyz = int_xyz.clip(self.entries_min[:, None, None, None], self.entries_num[:, None, None, None]-1)
        off_xyz = flt_xyz - int_xyz[:, :, 0]  # L, N, 3

        sh = self.start_hash
        nl = self.n_levels

        ind_dense: torch.Tensor = \
            int_xyz[:sh, ..., 0] * (self.entries_num[:sh]**2)[:, None, None] + \
            int_xyz[:sh, ..., 1] * (self.entries_num[:sh])[:, None, None] + \
            int_xyz[:sh, ..., 2]
        if self.separate_dense:
            ind_dense[1:] = ind_dense[1:] + self.entries_sum[:self.start_hash-1][:, None, None]

        ind_hash: torch.Tensor = (
            int_xyz[sh:, ..., 0]*cfg.ps[0] ^
            int_xyz[sh:, ..., 1]*cfg.ps[1] ^
            int_xyz[sh:, ..., 2]*cfg.ps[2]
        ) % self.n_entries_per_level

        if not self.separate_dense:
            ind = torch.cat([ind_dense, ind_hash], dim=0)

        L, T, F = self.n_levels, self.n_entries_per_level, self.f
        S, H = self.start_hash, self.n_levels - self.start_hash

        if self.separate_dense:
            val_dense = self.dense.gather(dim=0, index=ind_dense.view(S * N * 8)[..., None].expand(-1, F)).view(S, N, 8, F)
            val_hash = self.hash.gather(dim=1, index=ind_hash.view(H, N * 8)[..., None].expand(-1, -1, F)).view(H, N, 8, F)
            val = torch.cat([val_dense, val_hash], dim=0)
        else:
            val = self.hash.gather(dim=1, index=ind.view(L, N * 8)[..., None].expand(-1, -1, F)).view(L, N, 8, F)

        mul_xyz = (1 - self.offsets[None, None].to(self.device)) + (2 * self.offsets[None, None].to(self.device) - 1.) * off_xyz[:, :, None]
        mul_xyz = mul_xyz[..., 0] * mul_xyz[..., 1] * mul_xyz[..., 2]  # L, N, 8
        val = (mul_xyz[..., None] * val).sum(dim=-2)  # trilinear interpolated feature, L, N, F

        val = val.permute(1, 0, 2)  # N, L, F
        if self.sum:
            if self.sum_over_features:
                val = val.sum(dim=-1)
            else:
                val = val.sum(dim=-2)
        else:
            val = val.reshape(-1, L * F)

        if self.include_input:
            val = torch.cat([xyz, val], dim=-1)

        return val



if __name__ == "__main__":
    torch.manual_seed(0)
    xyz = torch.Tensor(
        [
            [-0.2, 0.4, 0.3],
            [0.3, -0.7, -0.3]
        ]
    ).cuda()
    batch = {
        'frame_dim': [0]
    }
    embedder = Embedder()
    print(embedder(xyz, batch))
