import os
import multiprocessing

_cpu_count = multiprocessing.cpu_count()
_physical_cores = max(1, _cpu_count // 2) if _cpu_count > 2 else max(1, _cpu_count)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def configure_torch(mode: str = "gpu_train") -> dict:
    _MODES = {
        "gpu_train": {"omp": _clamp(_physical_cores // 2, 2, 4), "mkl": _clamp(_physical_cores // 2, 2, 4), "intra_op": _clamp(_physical_cores // 2, 2, 4)},
        "gpu_infer": {"omp": 2, "mkl": 2, "intra_op": 2},
        "cpu_train": {"omp": _clamp(int(_physical_cores * 0.75), 2, _physical_cores), "mkl": _clamp(int(_physical_cores * 0.75), 2, _physical_cores), "intra_op": _clamp(int(_physical_cores * 0.75), 2, _physical_cores)},
        "cpu_infer": {"omp": _clamp(int(_physical_cores * 0.75), 2, _physical_cores), "mkl": _clamp(int(_physical_cores * 0.75), 2, _physical_cores), "intra_op": _clamp(int(_physical_cores * 0.75), 2, _physical_cores)},
    }
    cfg = _MODES.get(mode, _MODES["gpu_train"])

    omp_threads = cfg["omp"]
    mkl_threads = cfg["mkl"]

    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(mkl_threads))

    result = {
        "omp_threads": omp_threads,
        "mkl_threads": mkl_threads,
        "pin_memory": True,
        "intra_op_threads": cfg["intra_op"],
    }

    try:
        import torch
        torch.set_num_threads(cfg["intra_op"])
        result["intra_op_threads"] = torch.get_num_threads()
    except ImportError:
        pass

    return result


def get_dataloader_config(mode: str = "gpu_train", dataset_size: int = 0) -> dict:
    if dataset_size < 100:
        num_workers = 0
    elif dataset_size < 500:
        num_workers = 2
    else:
        num_workers = min(4, _physical_cores)
    return {
        "num_workers": num_workers,
        "pin_memory": True,
    }


def get_non_blocking() -> bool:
    return True


def worker_init_fn(worker_id: int):
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
