import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import argparse
import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib
import itertools
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def k2i_torch(K, ax=[-3, -2, -1]):
    X = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(K, dim=ax), dim=ax, norm="ortho"),
                           dim=ax)
    return X


def i2k_torch(K, ax=[-3, -2, -1]):
    X = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(K, dim=ax), dim=ax, norm="ortho"),
                           dim=ax)
    return X

def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def shrink1(s, alph, p, ep=1e-10):
    t = torch.abs(s)
    w = torch.max(t - alph * (t ** 2 + ep) ** (p / 2 - 0.5), torch.tensor(0.0, device=s.device)) * s 
    t[t == 0] = 1
    w = w / t
    return w


def SoftThres(X, reg):
    X = torch.sgn(X) * (torch.abs(X) - reg) * ((torch.abs(X) - reg) > 0)
    return X

def Sparse(S, reg, ax=(0, 1), HADAMARD=0):
    if HADAMARD:
        S1 = S * 0
        S1[0] = (-S[0] - S[1] - S[2] - S[3])/2.
        S1[1] = (-S[0] - S[1] + S[2] + S[3])/2.
        S1[2] = (-S[0] + S[1] - S[2] + S[3])/2.
        S1[3] = (-S[0] + S[1] + S[2] - S[3])/2.
        S = S1
    WX = i2k_torch(S, ax=ax)
    temp = SoftThres(WX, reg)
    if HADAMARD:
        temp1 = temp * 0
        temp1[0] = (-temp[0] - temp[1] - temp[2] - temp[3])/2.
        temp1[1] = (-temp[0] - temp[1] + temp[2] + temp[3])/2.
        temp1[2] = (-temp[0] + temp[1] - temp[2] + temp[3])/2.
        temp1[3] = (-temp[0] + temp[1] + temp[2] - temp[3])/2.
        temp = temp1
    return k2i_torch(temp, ax=ax), torch.sum(torch.abs(WX)).item()

def GETWIDTH(M, N, B):
    temp = (np.sqrt(M) + np.sqrt(N))
    if M > N:
        return temp + np.sqrt(np.log(B * N))
    else:
        return temp + np.sqrt(np.log(B * M))


def SVT(X, reg):
    Nv, Nt, FE, PE, SPE = X.shape
    U, S, Vh = torch.linalg.svd(X.view(Nv, Nt, -1), full_matrices=False)
    S_new = SoftThres(S, reg)
    S_new = torch.diag_embed(S_new).to(torch.complex64)
    X = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Nv, Nt, FE, PE, SPE)
    return X, torch.sum(torch.abs(S_new)).item()


def SVT_LLR(X, reg, blk):
    Nv, Nt, FE, PE, SPE = X.shape
    for i in range(Nv):
        Xi = X[i].clone()
        stepx = np.ceil(FE / blk)
        stepy = np.ceil(PE / blk)
        stepz = np.ceil(SPE / blk)
        padx = (stepx * blk).astype('int64')
        pady = (stepy * blk).astype('int64')
        padz = (stepz * blk).astype('int64')
        rrx = torch.randperm(blk)[0]
        rry = torch.randperm(blk)[0]
        rrz = torch.randperm(blk)[0]
        Xi = F.pad(Xi, (0, padz - SPE, 0, pady - PE, 0, padx - FE))
        Xi = torch.roll(Xi, (rrz, rry, rrx), (-1, -2, -3))
        FEp, PEp, SPEp = Xi.shape[-3:]
        patches = Xi.unfold(1, blk, blk).unfold(2, blk, blk).unfold(3, blk, blk)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(Nt, -1, blk, blk, blk).permute(1, 0, 2, 3, 4)
        Nb = patches.shape[0]
        U, S, Vh = torch.linalg.svd(patches.view(Nb, Nt, -1), full_matrices=False)
        S_new = SoftThres(S, reg)
        S_new = torch.diag_embed(S_new).to(torch.complex64)
        patches = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Nb, Nt, blk, blk, blk)
        patches = patches.permute((1, 0, 2, 3, 4))
        patches_orig = patches.view(unfold_shape)
        patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        patches_orig = patches_orig.view(Nt, FEp, PEp, SPEp)
        patches_orig = torch.roll(patches_orig, (-rrz, -rry, -rrx), (-1, -2, -3))
        X[i] = patches_orig[..., :FE, :PE, :SPE]
    return X, torch.sum(torch.abs(S_new)).item()

def HAAR4D(X, scale=1 / 2, forward=True, device='cuda'):
    if forward:
        Nv, Nt, FE, PE, SPE = X.shape
        sign = [-1, 1]
        sign_comb = torch.tensor(list(itertools.product([-1, 1], repeat=4)), device=device)
        d1, d2, d3, d4 = sign_comb.unbind(dim=1)
        X = X.unsqueeze(0)  
        d1 = d1.view(-1, 1, 1, 1, 1, 1) 
        d2 = d2.view(-1, 1, 1, 1, 1, 1) 
        d3 = d3.view(-1, 1, 1, 1, 1, 1) 
        d4 = d4.view(-1, 1, 1, 1, 1, 1) 
        tmp = scale * (X + d1 * torch.roll(X, shifts=-1, dims=2))
        tmp1 = scale * (tmp + d2 * torch.roll(tmp, shifts=-1, dims=3))
        tmp = scale * (tmp1 + d3 * torch.roll(tmp1, shifts=-1, dims=4))
        y = scale * (tmp + d4 * torch.roll(tmp, shifts=-1, dims=5))
        return y
    else:
        Ch, Nv, Nt, FE, PE, SPE = X.shape
        sign_comb = torch.tensor(list(itertools.product([-1, 1], repeat=4)), device=device)
        d1, d2, d3, d4 = sign_comb.unbind(dim=1) 
        d1 = d1.view(-1, 1, 1, 1, 1, 1) 
        d2 = d2.view(-1, 1, 1, 1, 1, 1) 
        d3 = d3.view(-1, 1, 1, 1, 1, 1) 
        d4 = d4.view(-1, 1, 1, 1, 1, 1) 

        tmp = scale * (X + d1 * torch.roll(X, shifts=1, dims=2))
        tmp1 = scale * (tmp + d2 * torch.roll(tmp, shifts=1, dims=3))
        tmp = scale * (tmp1 + d3 * torch.roll(tmp1, shifts=1, dims=4))
        y = scale * (tmp + d4 * torch.roll(tmp, shifts=1, dims=5))
        y = torch.sum(y, 0) 
        return y


def ST_HAAR(X, reg_list, device):
    temp = HAAR4D(X, forward=True, device=device)
    loss = 0
    for i in range(temp.shape[0]):
        temp[i:i+1] = SoftThres(temp[i:i+1], reg_list[i])
        loss += (torch.sum(torch.abs(temp[i:i+1])) * reg_list[i]).item()
    return HAAR4D(temp,forward=False, device=device), loss
    
def make_mask(usv, t, PE, SPE):
    ng1, ng2 = np.meshgrid(np.linspace(-1, 1, PE), np.linspace(-1, 1, SPE), indexing='ij')
    v = np.sqrt(ng1 ** 2 + ng2 ** 2)
    v = np.reshape(v, [1, PE, SPE])
    v = v / np.max(v)
    masks = np.random.uniform(size=[t, PE, SPE]) > v ** usv
    masks[:, PE // 2, SPE // 2] = 1.
    # Nt FE PE SPE
    return np.expand_dims(masks, axis=((0, 1, 3, 4)))


class Eop():
    def __init__(self, csm, us_mask):
        super(Eop, self).__init__()
        self.csm = csm
        self.us_mask = us_mask

    def mtimes(self, b, inv):
        if inv:
            x = torch.sum(k2i_torch(b * self.us_mask, ax=[-3, -2, -1]) * torch.conj(self.csm), dim=-4)
        else:
            b = b.unsqueeze(-4) * self.csm
            x = i2k_torch(b, ax=[-3, -2, -1]) * self.us_mask
        return x