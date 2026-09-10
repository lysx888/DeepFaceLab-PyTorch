"""DFL 等价优化器（PyTorch 原生实现）。

含 AdaBelief、RMSprop，均支持 lr_dropout / lr_cos / clipnorm，
法与 DFL leras/optimizers 完全对齐。
lr_dropout：对齐 DFL 语义 —— 每个参数在状态首次初始化时采样一次固定 bernoulli mask
（DFL 在 initialize_variables 时用 random_binomial 采样一次，之后每个 step 复用同一 mask），
mask 存入优化器 state（state_dict 自动序列化/恢复）。
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
    torch.nn.utils.clip_grad_norm_(params, max_norm=clipnorm, norm_type=2.0)


def _get_lr_dropout_mask(p, lr_dropout, state):
    """返回固定 lr_dropout mask（存于优化器 state，会话内不变）。

    对齐 DFL：initialize_variables 时 random_binomial 采样一次，之后每步复用。
    mask 存入 state['lr_mask']，随 state_dict 保存/恢复；全新状态时惰性采样一次。
    """
    if lr_dropout == 1.0:
        return None
    mask = state.get('lr_mask')
    if mask is None:
        mask = torch.bernoulli(torch.full_like(p, lr_dropout))
        state['lr_mask'] = mask
    return mask


def _apply_lr_cos(lr, lr_cos, iteration):
    if lr_cos != 0:
        return lr * (math.cos(iteration * (2 * math.pi / float(lr_cos))) + 1.0) / 2.0
    return lr


def _apply_lr_anneal(lr, lr_anneal, iteration, lr_anneal_start=0):
    """单调下降余弦退火：lr → lr*0.01，T_max步后保持lr*0.01。
    lr_anneal_start: 退火起始迭代（支持中途开启ca从满lr开始）。"""
    if lr_anneal > 0:
        lr_min = lr * 0.01
        progress = min(max(iteration - lr_anneal_start, 0) / float(lr_anneal), 1.0)
        return lr_min + (lr - lr_min) * (1.0 + math.cos(math.pi * progress)) / 2.0
    return lr


class _IterationPersistentMixin:
    """将 self._iteration 纳入 state_dict/load_state_dict，续训后 lr_cos 相位不丢。"""

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd['_iteration'] = self._iteration
        return sd

    def load_state_dict(self, state_dict, *args, **kwargs):
        self._iteration = state_dict.pop('_iteration', 0)
        super().load_state_dict(state_dict, *args, **kwargs)


class AdaBelief(_IterationPersistentMixin, Optimizer):
    """AdaBelief 优化器，对齐 DFL leras/optimizers/AdaBelief.py。

    v_t = β₂*v + (1-β₂)*(g - m_t)²  ← AdaBelief 核心（非 g²）
    无 bias correction（与 DFL 一致）。
    """

    def __init__(self, params, lr=5e-5, betas=(0.9, 0.999), eps=None,
                 weight_decay=0.0, lr_dropout=1.0, lr_cos=0, lr_anneal=0,
                 lr_anneal_start=0, clipnorm=0.0):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        if lr_dropout >= 1.0:
            lr_cos = 0
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lr_dropout=lr_dropout, lr_cos=lr_cos, lr_anneal=lr_anneal,
                        lr_anneal_start=lr_anneal_start, clipnorm=clipnorm)
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
            lr_anneal = group['lr_anneal']
            lr_anneal_start = group['lr_anneal_start']
            clipnorm = group['clipnorm']

            _global_clip_grad_norm(group['params'], clipnorm)
            lr = _apply_lr_anneal(lr, lr_anneal, self._iteration, lr_anneal_start)
            lr = _apply_lr_cos(lr, lr_cos, self._iteration)

            ps = [p for p in group['params'] if p.grad is not None]
            if not ps:
                continue

            # 状态初始化（zeros_like 不耗 RNG，顺序与逐参数实现一致）
            for p in ps:
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

            # foreach 批量实现：kernel 从 per-param 降到每层几次
            grads = [p.grad for p in ps]
            if wd != 0:
                grads = torch._foreach_add(grads, ps, alpha=wd)  # 非原地，不污染 p.grad

            ms = [self.state[p]['m'] for p in ps]
            vs = [self.state[p]['v'] for p in ps]

            torch._foreach_mul_(ms, beta1)
            torch._foreach_add_(ms, grads, alpha=1.0 - beta1)

            diffs = torch._foreach_sub(grads, ms)  # 非原地
            torch._foreach_pow_(diffs, 2)

            torch._foreach_mul_(vs, beta2)
            torch._foreach_add_(vs, diffs, alpha=1.0 - beta2)

            denoms = torch._foreach_sqrt(vs)  # 非原地
            cur_eps_list = [_dtype_resolution(g.dtype) if eps is None else eps for g in grads]
            torch._foreach_add_(denoms, cur_eps_list)

            updates = torch._foreach_div(ms, denoms)  # 非原地
            torch._foreach_mul_(updates, -lr)

            if lr_dropout != 1.0:
                masks = [_get_lr_dropout_mask(p, lr_dropout, self.state[p]) for p in ps]
                torch._foreach_mul_(updates, masks)

            torch._foreach_add_(ps, updates)

        return loss


class RMSprop(_IterationPersistentMixin, Optimizer):
    """RMSprop 优化器，对齐 DFL leras/optimizers/RMSprop.py。"""

    def __init__(self, params, lr=5e-5, rho=0.9, eps=None,
                 lr_dropout=1.0, lr_cos=0, lr_anneal=0,
                 lr_anneal_start=0, clipnorm=0.0):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        if lr_dropout >= 1.0:
            lr_cos = 0
        defaults = dict(lr=lr, rho=rho, eps=eps,
                        lr_dropout=lr_dropout, lr_cos=lr_cos, lr_anneal=lr_anneal,
                        lr_anneal_start=lr_anneal_start, clipnorm=clipnorm)
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
            lr_anneal = group['lr_anneal']
            lr_anneal_start = group['lr_anneal_start']
            clipnorm = group['clipnorm']

            _global_clip_grad_norm(group['params'], clipnorm)
            lr = _apply_lr_anneal(lr, lr_anneal, self._iteration, lr_anneal_start)
            lr = _apply_lr_cos(lr, lr_cos, self._iteration)

            ps = [p for p in group['params'] if p.grad is not None]
            if not ps:
                continue

            # 状态初始化（顺序与逐参数实现一致）
            for p in ps:
                state = self.state[p]
                if len(state) == 0:
                    state['a'] = torch.zeros_like(p)

            # foreach 批量实现
            grads = [p.grad for p in ps]
            as_ = [self.state[p]['a'] for p in ps]

            g_sqs = torch._foreach_pow(grads, 2)  # 非原地
            torch._foreach_mul_(as_, rho)
            torch._foreach_add_(as_, g_sqs, alpha=1.0 - rho)

            denoms = torch._foreach_sqrt(as_)  # 非原地
            cur_eps_list = [_dtype_resolution(g.dtype) if eps is None else eps for g in grads]
            torch._foreach_add_(denoms, cur_eps_list)

            updates = torch._foreach_div(grads, denoms)  # 非原地
            torch._foreach_mul_(updates, -lr)

            if lr_dropout != 1.0:
                masks = [_get_lr_dropout_mask(p, lr_dropout, self.state[p]) for p in ps]
                torch._foreach_mul_(updates, masks)

            torch._foreach_add_(ps, updates)

        return loss


class Adam(_IterationPersistentMixin, Optimizer):
    """Adam 优化器（标准实现，含 bias correction）。

    m_t = β₁*m + (1-β₁)*g
    v_t = β₂*v + (1-β₂)*g²
    update = -lr * m_t/(1-β₁^t) / (√(v_t/(1-β₂^t)) + eps)
    """

    def __init__(self, params, lr=5e-5, betas=(0.9, 0.999), eps=None,
                 weight_decay=0.0, lr_dropout=1.0, lr_cos=0, lr_anneal=0,
                 lr_anneal_start=0, clipnorm=0.0):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        if lr_dropout >= 1.0:
            lr_cos = 0
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lr_dropout=lr_dropout, lr_cos=lr_cos, lr_anneal=lr_anneal,
                        lr_anneal_start=lr_anneal_start, clipnorm=clipnorm)
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
            lr_anneal = group['lr_anneal']
            lr_anneal_start = group['lr_anneal_start']
            clipnorm = group['clipnorm']

            _global_clip_grad_norm(group['params'], clipnorm)
            lr = _apply_lr_anneal(lr, lr_anneal, self._iteration, lr_anneal_start)
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

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                m_hat = m / bias_c1
                v_hat = v / bias_c2
                update = m_hat.div_(v_hat.sqrt_().add_(cur_eps))
                update.mul_(-lr)

                mask = _get_lr_dropout_mask(p, lr_dropout, state)
                if mask is not None:
                    update.mul_(mask)

                p.add_(update)

        return loss


class AdamW(_IterationPersistentMixin, Optimizer):
    """AdamW 优化器（decoupled weight decay，含 bias correction）。

    m_t = β₁*m + (1-β₁)*g
    v_t = β₂*v + (1-β₂)*g²
    update = -lr * m_t/(1-β₁^t) / (√(v_t/(1-β₂^t)) + eps)
    p = p - lr * wd * p  (decoupled weight decay)
    """

    def __init__(self, params, lr=5e-5, betas=(0.9, 0.999), eps=None,
                 weight_decay=0.0, lr_dropout=1.0, lr_cos=0, lr_anneal=0,
                 lr_anneal_start=0, clipnorm=0.0):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        if lr_dropout >= 1.0:
            lr_cos = 0
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lr_dropout=lr_dropout, lr_cos=lr_cos, lr_anneal=lr_anneal,
                        lr_anneal_start=lr_anneal_start, clipnorm=clipnorm)
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
            lr_anneal = group['lr_anneal']
            lr_anneal_start = group['lr_anneal_start']
            clipnorm = group['clipnorm']

            _global_clip_grad_norm(group['params'], clipnorm)
            lr = _apply_lr_anneal(lr, lr_anneal, self._iteration, lr_anneal_start)
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

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                m_hat = m / bias_c1
                v_hat = v / bias_c2
                update = m_hat.div_(v_hat.sqrt_().add_(cur_eps))
                update.mul_(-lr)

                if wd != 0:
                    update.add_(p, alpha=-lr * wd)

                mask = _get_lr_dropout_mask(p, lr_dropout, state)
                if mask is not None:
                    update.mul_(mask)

                p.add_(update)

        return loss
