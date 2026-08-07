import os

DEBUG_TRACE = os.environ.get('SAEHD_DEBUG', '0') == '1'

def tlog(tag: str, **kwargs):
    if not DEBUG_TRACE:
        return
    parts = []
    for k, v in kwargs.items():
        if hasattr(v, 'shape') and hasattr(v, 'dtype'):
            import torch
            with torch.no_grad():
                vf = v.detach().float()
                parts.append(f"{k}=[shape={tuple(v.shape)} dtype={v.dtype} "
                             f"mean={vf.mean():.6f} min={vf.min():.6f} max={vf.max():.6f} "
                             f"std={vf.std():.6f}]")
        elif isinstance(v, (int, float, bool, str)):
            parts.append(f"{k}={v}")
        elif v is None:
            parts.append(f"{k}=None")
        else:
            parts.append(f"{k}=type({type(v).__name__})")
    print(f"[TRACE:{tag}] {' | '.join(parts)}", flush=True)

def tlog_grad(tag: str, named_params):
    if not DEBUG_TRACE:
        return
    import torch
    for name, p in named_params:
        if p.grad is not None:
            g = p.grad.detach().float()
            tlog(f"{tag}.grad", param=name, grad_mean=g.mean(), grad_abs_max=g.abs().max())
