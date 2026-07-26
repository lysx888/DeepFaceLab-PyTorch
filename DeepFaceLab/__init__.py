from DeepFaceLab.shared.torch_config import configure_torch, get_dataloader_config, get_non_blocking
configure_torch("gpu_train")
__version__ = "0.1.0"
