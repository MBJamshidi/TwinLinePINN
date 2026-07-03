"""
pinn_core.py
============
A from-scratch PINN for the telegrapher equations, in pure NumPy.

The PDE residuals require ∂u/∂x and ∂u/∂t of the neural network.
We propagate the input-Jacobian J = ∂u/∂(x,t) forward through the layers
analytically and backprop through both u and J in one pass.

Network: tanh-MLP, Glorot init, linear output.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ============================================================
@dataclass
class MLP:
    layers: List[int]
    seed: int = 0
    Ws: List[np.ndarray] = field(default_factory=list)
    bs: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        for n_in, n_out in zip(self.layers[:-1], self.layers[1:]):
            std = np.sqrt(2.0 / (n_in + n_out))
            self.Ws.append(rng.normal(0.0, std, size=(n_in, n_out)))
            self.bs.append(np.zeros(n_out))

    @property
    def n_params(self):
        return sum(W.size for W in self.Ws) + sum(b.size for b in self.bs)

    def get_params(self):
        return np.concatenate([W.ravel() for W in self.Ws]
                              + [b.ravel() for b in self.bs])

    def set_params(self, vec):
        idx = 0
        for k, W in enumerate(self.Ws):
            n = W.size
            self.Ws[k] = vec[idx:idx+n].reshape(W.shape).copy(); idx += n
        for k, b in enumerate(self.bs):
            n = b.size
            self.bs[k] = vec[idx:idx+n].copy(); idx += n


# ============================================================
def forward(net: MLP, X: np.ndarray):
    """Plain forward, returns u and a layer cache for backprop."""
    A = X
    cache = []
    L = len(net.Ws)
    for k in range(L):
        Z = A @ net.Ws[k] + net.bs[k]
        A_next = np.tanh(Z) if k < L - 1 else Z
        cache.append(dict(A_in=A, Z=Z, A_out=A_next))
        A = A_next
    return A, cache


def forward_with_jac(net: MLP, X: np.ndarray):
    """Forward + input-Jacobian J = ∂u/∂X (shape (N, d_out, d_in)).

    For a tanh-MLP with linear output:
        z   = A_in W + b
        a   = tanh(z)            (hidden)
        J_a = (1 - a^2) * (J_in @ W)        (broadcast over batch)
    For the final linear layer:
        a   = z
        J   = J_in @ W
    Each cache entry stores J_in to make the backward pass O(L).
    """
    A = X                                                       # (N, d_in)
    N, d_in = X.shape
    J = np.broadcast_to(np.eye(d_in), (N, d_in, d_in)).copy()   # (N, d_out_curr=d_in, d_in)
    cache = []
    L = len(net.Ws)
    for k in range(L):
        W = net.Ws[k]
        Z = A @ W + net.bs[k]
        J_in = J
        Jpre = J @ W                              # (N, d_in, n_out)
        if k < L - 1:
            A_next = np.tanh(Z)
            phi_p = 1.0 - A_next ** 2             # (N, n_out)
            J = phi_p[:, None, :] * Jpre          # broadcast on output dim
        else:
            A_next = Z
            J = Jpre
        cache.append(dict(A_in=A, Z=Z, A_out=A_next, W=W, J_in=J_in))
        A = A_next
    # J shape: (N, d_in, d_out_last); we transpose at use sites if needed.
    return A, J, cache


# ============================================================
@dataclass
class Problem:
    Rp: float; Lp: float; Gp: float; Cp: float
    length: float; T: float
    V_star: float = 1.0

    @property
    def Z0(self):     return float(np.sqrt(self.Lp / self.Cp))
    @property
    def c(self):      return float(1.0 / np.sqrt(self.Lp * self.Cp))
    @property
    def I_star(self): return self.V_star / self.Z0


# ============================================================
def grad_supervised(net: MLP, X: np.ndarray, target: np.ndarray):
    """Gradient of (1/N) Σ ||u(X) - target||² wrt all parameters."""
    u, cache = forward(net, X)
    err = u - target                              # (N, d_out)
    delta = (2.0 / X.shape[0]) * err
    L = len(net.Ws)
    gW = [np.zeros_like(W) for W in net.Ws]
    gb = [np.zeros_like(b) for b in net.bs]
    for k in reversed(range(L)):
        c = cache[k]
        if k < L - 1:
            delta = delta * (1.0 - c["A_out"] ** 2)
        gW[k] += c["A_in"].T @ delta
        gb[k] += delta.sum(axis=0)
        delta = delta @ net.Ws[k].T
    loss = float(np.mean(np.sum(err ** 2, axis=1)))
    flat = np.concatenate([g.ravel() for g in gW] + [g.ravel() for g in gb])
    return loss, flat


def _backprop_pde(net: MLP, X: np.ndarray, grad_u: np.ndarray,
                  grad_J: np.ndarray, cache, J_final):
    """
    Backprop for a loss with sensitivities (grad_u, grad_J) wrt the
    network output u and the input-Jacobian J.

    grad_u : (N, d_out)
    grad_J : (N, d_in, d_out)   matching forward_with_jac convention
    cache  : output of forward_with_jac
    """
    L = len(net.Ws)
    gW = [np.zeros_like(W) for W in net.Ws]
    gb = [np.zeros_like(b) for b in net.bs]
    adj_a = grad_u.copy()
    adj_J = grad_J.copy()
    N = X.shape[0]; d_in = X.shape[1]

    for k in reversed(range(L)):
        c = cache[k]
        A_in = c["A_in"]; a = c["A_out"]; J_in = c["J_in"]; W = net.Ws[k]
        n_in_k, n_out_k = W.shape

        if k == L - 1:                            # linear layer
            gW[k] += A_in.T @ adj_a
            gb[k] += adj_a.sum(axis=0)
            # gW += sum_n J_in[n,:,:].T @ adj_J[n,:,:]   (over the d_in axis)
            J_flat   = J_in.reshape(N * d_in, n_in_k)
            adjJ_flat = adj_J.reshape(N * d_in, n_out_k)
            gW[k] += J_flat.T @ adjJ_flat
            adj_A_in = adj_a @ W.T
            adj_J_in = adj_J @ W.T                # (N, d_in, n_in_k)
        else:                                     # tanh layer
            phi_p   = 1.0 - a ** 2                # (N, n_out)
            phi_pp  = -2.0 * a * phi_p
            JinW    = J_in @ W                    # (N, d_in, n_out)
            adj_z_direct = adj_a * phi_p
            adj_z_jacpath = (adj_J * JinW * phi_pp[:, None, :]).sum(axis=1)
            adj_z = adj_z_direct + adj_z_jacpath
            gW[k] += A_in.T @ adj_z
            gb[k] += adj_z.sum(axis=0)
            scaled = adj_J * phi_p[:, None, :]
            J_flat = J_in.reshape(N * d_in, n_in_k)
            scaled_flat = scaled.reshape(N * d_in, n_out_k)
            gW[k] += J_flat.T @ scaled_flat
            adj_A_in = adj_z @ W.T
            adj_J_in = scaled @ W.T

        adj_a = adj_A_in
        adj_J = adj_J_in

    return np.concatenate([g.ravel() for g in gW] + [g.ravel() for g in gb])


# ============================================================
def total_loss_and_grad(net, prob: Problem, sets: Dict,
                        weights=(1e-2, 10.0, 10.0, 5.0),
                        phi_override: Optional[Tuple[float, float, float, float]] = None):
    """
    Compute total loss and its gradient wrt theta.

    sets keys:
        coll       (Np,2)  collocation points in NORMALIZED coords
        ic_pts, ic_vals      (Ni,2)  (Ni,2)
        bc_pts, bc_vals      (Nb,2)  (Nb,2)
        data_pts, data_vals  (Nd,2)  (Nd,2)   (may be empty)

    weights = (wp, wi, wb, wd)
    phi_override : if not None, replaces (Rp,Lp,Gp,Cp).
    """
    wp, wi, wb, wd = weights
    Rp = prob.Rp if phi_override is None else phi_override[0]
    Lp = prob.Lp if phi_override is None else phi_override[1]
    Gp = prob.Gp if phi_override is None else phi_override[2]
    Cp = prob.Cp if phi_override is None else phi_override[3]

    Vs_, Is_ = prob.V_star, prob.I_star
    Lx, Tt   = prob.length, prob.T

    # --- PDE residuals ---
    coll = sets['coll']
    Np   = coll.shape[0]
    u_n, J_n, cache_pde = forward_with_jac(net, coll)
    Vn  = u_n[:, 0];   In_  = u_n[:, 1]
    Vx  = (Vs_ / Lx) * J_n[:, 0, 0]
    Vt  = (Vs_ / Tt) * J_n[:, 1, 0]
    Ix  = (Is_ / Lx) * J_n[:, 0, 1]
    It  = (Is_ / Tt) * J_n[:, 1, 1]
    V_phys = Vs_ * Vn; I_phys = Is_ * In_
    rV = Vx + Lp * It + Rp * I_phys
    rI = Ix + Cp * Vt + Gp * V_phys
    Lpde = float(np.mean(rV ** 2 + rI ** 2))

    # gradients of L_pde wrt u and J (in normalized coords)
    twoN = 2.0 / Np
    grad_u_pde = np.zeros_like(u_n)
    grad_u_pde[:, 0] = twoN * rI * Gp * Vs_
    grad_u_pde[:, 1] = twoN * rV * Rp * Is_
    grad_J_pde = np.zeros_like(J_n)            # (N, d_in=2, d_out=2)
    # column 0 -> outputs[0]=V; J[:,k,0] = ∂Vn/∂xn (k=0) or ∂Vn/∂tn (k=1)
    grad_J_pde[:, 0, 0] = twoN * rV * (Vs_ / Lx)            # ∂L/∂(∂Vn/∂xn)
    grad_J_pde[:, 1, 0] = twoN * rI * Cp * (Vs_ / Tt)       # ∂L/∂(∂Vn/∂tn)
    grad_J_pde[:, 0, 1] = twoN * rI * (Is_ / Lx)            # ∂L/∂(∂In/∂xn)
    grad_J_pde[:, 1, 1] = twoN * rV * Lp * (Is_ / Tt)       # ∂L/∂(∂In/∂tn)
    g_pde = _backprop_pde(net, coll, grad_u_pde, grad_J_pde,
                          cache_pde, J_n)

    # --- Supervised parts ---
    g_total = wp * g_pde
    comps = dict(pde=Lpde, ic=0.0, bc=0.0, data=0.0)

    pieces = [('ic', sets['ic_pts'], sets['ic_vals'], wi),
              ('bc', sets['bc_pts'], sets['bc_vals'], wb),
              ('data', sets['data_pts'], sets['data_vals'], wd)]
    for name, X, U, w_ in pieces:
        if X is None or len(X) == 0 or w_ == 0.0:
            continue
        loss_, g_ = grad_supervised(net, X, U)
        comps[name] = loss_
        g_total += w_ * g_

    total = wp * Lpde + wi * comps['ic'] + wb * comps['bc'] + wd * comps['data']
    comps['total'] = total
    return total, g_total, comps, (rV, rI, u_n, J_n)


# ============================================================
@dataclass
class Adam:
    lr: float = 2e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    t: int = 0

    def step(self, p, g):
        if self.m is None:
            self.m = np.zeros_like(p); self.v = np.zeros_like(p)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2
        mhat = self.m / (1 - self.beta1 ** self.t)
        vhat = self.v / (1 - self.beta2 ** self.t)
        return p - self.lr * mhat / (np.sqrt(vhat) + self.eps)
