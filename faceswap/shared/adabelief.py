import math

import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    r"""AdaBelief optimizer: adaptive learning rate with belief in gradient direction.

    Key difference from Adam: second moment tracks E[(g - m)^2] instead of E[g^2].
    This gives larger steps when gradients are consistent (high SNR) and smaller
    steps when gradients oscillate (low SNR).

    Args:
        params: iterable of parameters
        lr: learning rate (default: 1e-3)
        betas: coefficients for running averages (default: (0.9, 0.999))
        eps: term added for numerical stability (default: 1e-8)
        weight_decay: L2 penalty (default: 0)
        amsgrad: use AMSGrad variant (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-5,
                 weight_decay=0, amsgrad=False, bias_correction=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, bias_correction=bias_correction)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                    state_steps.append(state["step"])

            self._adabelief(
                params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                state_steps, amsgrad=group["amsgrad"], bias_correction=group["bias_correction"],
                beta1=group["betas"][0],
                beta2=group["betas"][1], lr=group["lr"], weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    @staticmethod
    def _adabelief(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                   state_steps, amsgrad, bias_correction, beta1, beta2, lr, weight_decay, eps):
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            step += 1

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            grad_residual = grad - exp_avg
            exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

            if bias_correction:
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                denom_sqrt = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
            else:
                step_size = lr
                denom_sqrt = exp_avg_sq.sqrt()

            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = max_exp_avg_sq.sqrt().add_(eps)
            else:
                denom = denom_sqrt.add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)
