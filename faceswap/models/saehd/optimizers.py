"""DFL 等价优化器（PyTorch 原生实现）。

含 AdaBelief、RMSprop，均支持 lr_dropout / lr_cos / clipnorm，
法与 DFL leras/optimizers 完全对齐。
lr_dropout 每步动态生成 bernoulli mask（对齐 DFL random_binomial 行为）。
"""
import math
import torch
from torch.optim import Optimizer


def _dtype_resolution(dtype: torch.dtype) -> float:
    eps = torch.finfo(dtype).eps
    precision = math.floor(-math.log10(eps))
    return 10.0 ** (-precision)


def _global_clip_grad_norm(params, clipnorm):
    if clipnorm <= 0.0:
        return
    total_norm_sq = 0.0
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad)
            total_norm_sq += p.grad.detach().float().pow(2).sum().item()
    if total_norm_sq == 0.0:
        return
    total_norm = math.sqrt(total_norm_sq)
    if total_norm > clipnorm:
        scale = clipnorm / total_norm
        for g in grads:
            g.mul_(scale)


def _get_lr_dropout_mask(p, lr_dropout):
    if lr_dropout == 1.0:
        return None
    return torch.bernoulli(torch.full_like(p, lr_dropout))


def _apply_lr_cos(lr, lr_cos, iteration):
    if lr_cos != 0:
        return lr * (math.cos(iteration * (2 * math.pi / float(lr_cos))) + 1.0) / 2.0
    return lr


class AdaBelief(Optimizer):
    """AdaBelief 优化器，对齐 DFL leras/optimizers/AdaBelief.py。

    v_t = β₂*v + (1-β₂)*(g - m_t)²  ← AdaBelief 核心（非 g²）
    无 bias correction（与 DFL 一致）。
    """

    def __init__(self, params, lr=5e-5, betas=(0.9, 0.999), eps=None,
                 weight_decay=0.0, lr_dropout=1.0, lr_cos=0, clipnorm=0.0):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm)
        super().__init__(params, defaults)
        self._iteration = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._iteration += 1

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            lr_dropout = group['lr_dropout']
            lr_cos = group['lr_cos']
            clipnorm = group['clipnorm']

            _global_clip_grad_norm(group['params'], clipnorm)
            lr = _apply_lr_cos(lr, lr_cos, self._iteration)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                cur_eps = _dtype_resolution(g.dtype) if eps is None else eps
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m, v = state['m'], state['v']

                if wd != 0:
                    g = g.add(p, alpha=wd)

                m_t = beta1 * m + (1.0 - beta1) * g
                v_t = beta2 * v + (1.0 - beta2) * (g - m_t).pow(2)

                update = -lr * m_t / (v_t.sqrt() + cur_eps)

                mask = _get_lr_dropout_mask(p, lr_dropout)
                if mask is not None:
                    update = update * mask

                p.add_(update)
                m.copy_(m_t)
                v.copy_(v_t)

        return loss


class RMSprop(Optimizer):
    """RMSprop 优化器，对齐 DFL leras/optimizers/RMSprop.py。"""

    def __init__(self, params, lr=5e-5, rho=0.9, eps=None,
                 lr_dropout=1.0, lr_cos=0, clipnorm=0.0):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, rho=rho, eps=eps,
                        lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm)
        super().__init__(params, defaults)
        self._iteration = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._iteration += 1

        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            lr_dropout = group['lr_dropout']
            lr_cos = group['lr_cos']
            clipnorm = group['clipnorm']

            _global_clip_grad_norm(group['params'], clipnorm)
            lr = _apply_lr_cos(lr, lr_cos, self._iteration)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                cur_eps = _dtype_resolution(g.dtype) if eps is None else eps
                state = self.state[p]
                if len(state) == 0:
                    state['a'] = torch.zeros_like(p)
                a = state['a']

                a_t = rho * a + (1.0 - rho) * g.pow(2)
                update = -lr * g / (a_t.sqrt() + cur_eps)

                mask = _get_lr_dropout_mask(p, lr_dropout)
                if mask is not None:
                    update = update * mask

                p.add_(update)
                a.copy_(a_t)

        return loss


class Adam(Optimizer):
    """Adam 优化器（标准实现，含 bias correction）。

    m_t = β₁*m + (1-β₁)*g
    v_t = β₂*v + (1-β₂)*g²
    update = -lr * m_t/(1-β₁^t) / (√(v_t/(1-β₂^t)) + eps)
    """

    def __init__(self, params, lr=5e-5, betas=(0.9, 0.999), eps=None,
                 weight_decay=0.0, lr_dropout=1.0, lr_cos=0, clipnorm=0.0):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm)
        super().__init__(params, defaults)
        self._iteration = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._iteration += 1

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            lr_dropout = group['lr_dropout']
            lr_cos = group['lr_cos']
            clipnorm = group['clipnorm']

            _global_clip_grad_norm(group['params'], clipnorm)
            lr = _apply_lr_cos(lr, lr_cos, self._iteration)

            bias_c1 = 1.0 - beta1 ** self._iteration
            bias_c2 = 1.0 - beta2 ** self._iteration

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                cur_eps = _dtype_resolution(g.dtype) if eps is None else eps
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m, v = state['m'], state['v']

                if wd != 0:
                    g = g.add(p, alpha=wd)

                m_t = beta1 * m + (1.0 - beta1) * g
                v_t = beta2 * v + (1.0 - beta2) * g.pow(2)

                m_hat = m_t / bias_c1
                v_hat = v_t / bias_c2
                update = -lr * m_hat / (v_hat.sqrt() + cur_eps)

                mask = _get_lr_dropout_mask(p, lr_dropout)
                if mask is not None:
                    update = update * mask

                p.add_(update)
                m.copy_(m_t)
                v.copy_(v_t)

        return loss


class AdamW(Optimizer):
    """AdamW 优化器（decoupled weight decay，含 bias correction）。

    m_t = β₁*m + (1-β₁)*g
    v_t = β₂*v + (1-β₂)*g²
    update = -lr * m_t/(1-β₁^t) / (√(v_t/(1-β₂^t)) + eps)
    p = p - lr * wd * p  (decoupled weight decay)
    """

    def __init__(self, params, lr=5e-5, betas=(0.9, 0.999), eps=None,
                 weight_decay=0.0, lr_dropout=1.0, lr_cos=0, clipnorm=0.0):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm)
        super().__init__(params, defaults)
        self._iteration = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._iteration += 1

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            lr_dropout = group['lr_dropout']
            lr_cos = group['lr_cos']
            clipnorm = group['clipnorm']

            _global_clip_grad_norm(group['params'], clipnorm)
            lr = _apply_lr_cos(lr, lr_cos, self._iteration)

            bias_c1 = 1.0 - beta1 ** self._iteration
            bias_c2 = 1.0 - beta2 ** self._iteration

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                cur_eps = _dtype_resolution(g.dtype) if eps is None else eps
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m, v = state['m'], state['v']

                m_t = beta1 * m + (1.0 - beta1) * g
                v_t = beta2 * v + (1.0 - beta2) * g.pow(2)

                m_hat = m_t / bias_c1
                v_hat = v_t / bias_c2
                update = -lr * m_hat / (v_hat.sqrt() + cur_eps)

                if wd != 0:
                    update = update - lr * wd * p

                mask = _get_lr_dropout_mask(p, lr_dropout)
                if mask is not None:
                    update = update * mask

                p.add_(update)
                m.copy_(m_t)
                v.copy_(v_t)

        return loss
