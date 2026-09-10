import torch


def generate_summary_text(
    model_name: str,
    iter_count: int,
    config_dict: dict,
    param_labels: dict[str, str],
    modules_dict: dict[str, torch.nn.Module] | None = None,
    device: torch.device | None = None,
) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append(f"  模型名称: {model_name}  |  当前迭代: {iter_count}")
    lines.append("=" * 50)
    lines.append("")

    if config_dict:
        lines.append("--- 训练参数 ---")
        def _disp_width(s):
            return sum(2 if '\u4e00' <= c <= '\u9fff' or c in '，。：；！？（）' else 1 for c in s)
        max_w = max((_disp_width(param_labels.get(k, k)) for k in config_dict), default=0)
        for k, v in config_dict.items():
            label = param_labels.get(k, k)
            if isinstance(v, bool):
                val_str = "是" if v else "否"
            elif isinstance(v, float):
                val_str = f"{v:.6g}"
            else:
                val_str = str(v)
            pad = max_w - _disp_width(label)
            lines.append(f"  {label}{' ' * pad}  {val_str}")
        lines.append("")

    if modules_dict:
        total_params = 0
        trainable_params = 0
        module_info = []
        for name, module in modules_dict.items():
            n = sum(p.numel() for p in module.parameters())
            t = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += n
            trainable_params += t
            frozen = n - t
            status = "训练中" if t > 0 else "已冻结"
            module_info.append((name, n, t, frozen, status))

        if module_info:
            lines.append("--- 模块状态 ---")
            name_w = max(len(n) for n, *_ in module_info)
            for name, n, t, frozen, status in module_info:
                detail = f"参数: {n:,}"
                if frozen > 0:
                    detail += f"  (训练: {t:,}  冻结: {frozen:,})"
                lines.append(f"  {name:<{name_w}}  {detail}  [{status}]")
            lines.append(f"  总计: {total_params:,} 参数  (训练: {trainable_params:,}  冻结: {total_params - trainable_params:,})")
            lines.append("")

    lines.append("--- 运行设备 ---")
    if device is not None:
        from faceswap.shared.config import get_device_name, get_device_memory_mb, is_gpu_device
        dev_name = get_device_name(device)
        if is_gpu_device(device):
            vram = get_device_memory_mb(device) / 1024
            idx = device.index if device.index is not None else 0
            lines.append(f"  设备: {dev_name}  |  显存: {vram:.1f}GB  |  索引: {idx}")
        else:
            lines.append(f"  设备: {dev_name}")
    else:
        lines.append("  设备: CPU")
    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)
