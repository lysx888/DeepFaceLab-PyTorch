import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from faceswap.shared.logger import get_logger

_logger = get_logger("base_model")


class BaseModel(ABC):
    _model_prefix: str = ""
    _param_labels: dict[str, str] = {}
    _config_filename: str = ""
    _MAX_BACKUPS = 10

    def __init__(self, config, model_dir: Path, device: torch.device):
        self.config = config
        self.model_dir = Path(model_dir)
        self.device = device
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._modules_dict: dict[str, nn.Module] = {}
        self._optimizers_dict: dict[str, torch.optim.Optimizer] = {}
        self._model_filename_list: list[tuple[object, str]] = []
        self._aux_state: dict[str, object] = {}

        self.build()
        self._apply_init()
        self.apply_freeze()
        self.build_optimizers()
        self.try_load()

    def _prefixed(self, name: str) -> str:
        if self._model_prefix:
            return f"{self._model_prefix}_{name}"
        return name

    @abstractmethod
    def build(self) -> None:
        ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> dict:
        ...

    @abstractmethod
    def compute_loss(self, batch_src: dict, batch_dst: dict, fw: dict) -> dict:
        ...

    @abstractmethod
    def get_preview_section_names(self) -> list[str]:
        ...

    @abstractmethod
    def generate_preview_data(self, src_dataset, dst_dataset,
                               src_indices: list[int],
                               dst_indices: list[int]) -> dict[str, list]:
        ...

    def register_module(self, name: str, module: nn.Module) -> None:
        module = module.to(self.device)
        self._modules_dict[name] = module
        self._model_filename_list.append((module, f'{name}.pth'))

    def register_optimizer(self, name: str, optimizer: torch.optim.Optimizer) -> None:
        self._optimizers_dict[name] = optimizer
        self._model_filename_list.append((optimizer, f'{name}.pth'))

    def register_aux_state(self, name: str, state: object) -> None:
        self._aux_state[name] = state

    def get_model_filename_list(self) -> list[tuple[object, str]]:
        return self._model_filename_list

    def onSave(self, tmp_dir: Path) -> None:
        for obj, filename in self.get_model_filename_list():
            path = tmp_dir / filename
            if isinstance(obj, nn.Module):
                sd = self._strip_compile_prefix(obj.state_dict())
                torch.save(sd, path)
            else:
                torch.save(obj.state_dict(), path)

    def get_modules_dict(self) -> dict[str, nn.Module]:
        return self._modules_dict

    def get_optimizers_dict(self) -> dict[str, torch.optim.Optimizer]:
        return self._optimizers_dict

    def get_aux_state(self) -> dict[str, object]:
        return self._aux_state

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        params = []
        for m in self._modules_dict.values():
            params.extend(p for p in m.parameters() if p.requires_grad)
        return params

    def _apply_init(self) -> None:
        pass

    def apply_freeze(self) -> None:
        pass

    def build_optimizers(self) -> None:
        pass

    def on_pretrain_override(self) -> None:
        pass

    def try_load(self) -> None:
        self.cleanup_stale_tmp_dirs()
        save_marker = self.model_dir / '.save_complete'
        if not save_marker.exists():
            bk_dir = self.model_dir / 'autobackups' / '01'
            if bk_dir.exists() and any(bk_dir.glob('*.pth')):
                _logger.warning("Incomplete save detected, restoring from latest backup...")
                self._restore_from_backup(bk_dir)
            else:
                _logger.warning("Incomplete save detected, no backup available. Starting fresh.")
        else:
            save_marker.unlink(missing_ok=True)

        ts_pth = self.model_dir / self._prefixed('training_state.json')
        if ts_pth.exists():
            try:
                ts = json.loads(ts_pth.read_text())
                self._aux_state['iter_count'] = ts.get('iter', 0)
                if 'loss_history' in ts:
                    self._aux_state['loss_history'] = ts['loss_history']
                saved_archi = ts.get('archi', None)
                current_archi = getattr(self.config, 'archi', None)
                if current_archi is not None and saved_archi is not None and saved_archi != current_archi:
                    _logger.warning(f"Architecture mismatch: saved={saved_archi}, current={current_archi}. "
                                    f"Skipping weight loading to avoid corruption.")
                    return
            except Exception:
                pass
        else:
            old_iter = self.model_dir / 'iter_count.json'
            if old_iter.exists():
                try:
                    self._aux_state['iter_count'] = json.loads(old_iter.read_text()).get('iter', 0)
                except Exception:
                    pass
            old_loss = self.model_dir / 'loss_history.json'
            if old_loss.exists():
                try:
                    self._aux_state['loss_history'] = json.loads(old_loss.read_text())
                except Exception:
                    pass

        for obj, filename in self.get_model_filename_list():
            pth = self.model_dir / filename
            if pth.exists():
                try:
                    state = torch.load(pth, map_location=self.device, weights_only=False)
                    if isinstance(obj, nn.Module):
                        state = self._strip_compile_prefix(state)
                        obj.load_state_dict(state)
                        _logger.info(f"Loaded {filename}")
                    else:
                        obj.load_state_dict(state)
                except Exception as e:
                    _logger.warning(f"Failed to load {filename}: {e}")

        rng_pth = self.model_dir / self._prefixed('rng_state.pth')
        if rng_pth.exists():
            try:
                rng_state = torch.load(rng_pth, map_location='cpu', weights_only=False)
                torch.random.set_rng_state(rng_state)
                _logger.info("Restored CPU RNG state")
            except Exception as e:
                _logger.warning(f"Failed to restore CPU RNG state: {e}")

        cuda_rng_pth = self.model_dir / self._prefixed('cuda_rng_state.pth')
        if cuda_rng_pth.exists() and torch.cuda.is_available():
            try:
                cuda_rng = torch.load(cuda_rng_pth, map_location='cpu', weights_only=False)
                torch.cuda.set_rng_state(cuda_rng)
                _logger.info("Restored CUDA RNG state")
            except Exception as e:
                _logger.warning(f"Failed to restore CUDA RNG state (device change?): {e}")

        ds_idx_pth = self.model_dir / self._prefixed('dataset_index.json')
        if ds_idx_pth.exists():
            try:
                self._aux_state['dataset_index'] = json.loads(ds_idx_pth.read_text())
                _logger.info("Restored dataset index")
            except Exception as e:
                _logger.warning(f"Failed to restore dataset index: {e}")

        gan_state_pth = self.model_dir / self._prefixed('gan_state.json')
        if gan_state_pth.exists():
            try:
                self._aux_state['progressive_gan_state'] = json.loads(gan_state_pth.read_text())
                _logger.info("Restored progressive GAN state")
            except Exception:
                pass

    def save(self, iter_count: int) -> None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(dir=str(self.model_dir), prefix=".save_"))
        try:
            self.onSave(tmp_dir)
            if hasattr(self.config, 'to_dict'):
                config_dict = self.config.to_dict()
                config_name = self._config_filename or "training_config.json"
                (tmp_dir / config_name).write_text(
                    json.dumps(config_dict, indent=2, ensure_ascii=False), encoding='utf-8')
            training_state = {'iter': iter_count}
            loss_hist = self._aux_state.get('loss_history')
            if loss_hist is not None:
                training_state['loss_history'] = loss_hist
            archi = getattr(self.config, 'archi', None)
            if archi is not None:
                training_state['archi'] = archi
            (tmp_dir / self._prefixed('training_state.json')).write_text(
                json.dumps(training_state))
            rng_state = torch.random.get_rng_state()
            torch.save(rng_state, tmp_dir / self._prefixed('rng_state.pth'))
            if torch.cuda.is_available():
                try:
                    cuda_rng = torch.cuda.get_rng_state()
                    torch.save(cuda_rng, tmp_dir / self._prefixed('cuda_rng_state.pth'))
                except Exception:
                    pass
            ds_idx = self._aux_state.get('dataset_index')
            if ds_idx is not None:
                (tmp_dir / self._prefixed('dataset_index.json')).write_text(json.dumps(ds_idx))
            gan_state = self._aux_state.get('progressive_gan_state')
            if gan_state is not None:
                (tmp_dir / self._prefixed('gan_state.json')).write_text(json.dumps(gan_state))
            summary_text = self.get_summary_text(iter_count)
            (tmp_dir / self._prefixed('summary.txt')).write_text(summary_text, encoding='utf-8')
            (tmp_dir / '.save_complete').touch()
            for f in sorted(tmp_dir.iterdir()):
                if f.name == '.save_complete':
                    continue
                target = self.model_dir / f.name
                os.replace(str(f), str(target))
            os.replace(str(tmp_dir / '.save_complete'), str(self.model_dir / '.save_complete'))
        except Exception:
            self._cleanup_tmp_dir(tmp_dir)
            raise
        finally:
            self._cleanup_tmp_dir(tmp_dir)

    @staticmethod
    def _cleanup_tmp_dir(tmp_dir: Path) -> None:
        if tmp_dir.exists():
            try:
                import shutil
                shutil.rmtree(str(tmp_dir))
            except Exception as e:
                _logger.warning(f"Failed to cleanup temp dir {tmp_dir}: {e}")

    def cleanup_stale_tmp_dirs(self) -> None:
        for p in self.model_dir.glob(".save_*"):
            if p.is_dir():
                try:
                    import shutil
                    shutil.rmtree(str(p))
                    _logger.info(f"Cleaned up stale temp dir: {p}")
                except Exception as e:
                    _logger.warning(f"Failed to cleanup stale temp dir {p}: {e}")

    @staticmethod
    def _strip_compile_prefix(state_dict: dict) -> dict:
        prefix = '_orig_mod.'
        if any(k.startswith(prefix) for k in state_dict.keys()):
            return {k[len(prefix):] if k.startswith(prefix) else k: v
                    for k, v in state_dict.items()}
        return state_dict

    @torch.no_grad()
    def merge(self, warped_dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclass must implement merge()")

    def get_summary_text(self, iter_count: int) -> str:
        from faceswap.core.summary_utils import generate_summary_text
        config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
        model_name = self._model_prefix or self.model_dir.name
        return generate_summary_text(
            model_name=model_name,
            iter_count=iter_count,
            config_dict=config_dict,
            param_labels=self._param_labels,
            modules_dict=self._modules_dict,
            device=self.device,
        )

    def _get_backup_files(self) -> list[Path]:
        backup_files = []
        for _, filename in self.get_model_filename_list():
            p = self.model_dir / filename
            if p.exists():
                backup_files.append(p)
        config_name = self._config_filename
        if config_name:
            p = self.model_dir / config_name
            if p.exists():
                backup_files.append(p)
        for fname in [self._prefixed('training_state.json'),
                      self._prefixed('summary.txt')]:
            p = self.model_dir / fname
            if p.exists():
                backup_files.append(p)
        return backup_files

    def create_backup(self, max_backups: int = 0) -> None:
        if max_backups <= 0:
            max_backups = self._MAX_BACKUPS
        backups_dir = self.model_dir / "autobackups"
        if not backups_dir.exists():
            backups_dir.mkdir(parents=True, exist_ok=True)

        backup_files = self._get_backup_files()
        if not backup_files:
            return

        for i in range(max_backups, 0, -1):
            idx_str = f'{i:02d}'
            idx_dir = backups_dir / idx_str
            next_dir = backups_dir / f'{i + 1:02d}'
            if not idx_dir.exists():
                continue
            if i == max_backups:
                shutil.rmtree(str(idx_dir), ignore_errors=True)
            else:
                try:
                    if next_dir.exists():
                        shutil.rmtree(str(next_dir))
                    shutil.move(str(idx_dir), str(next_dir))
                except Exception as e:
                    _logger.warning(f"Backup rotate {idx_str}→{i+1:02d} failed: {e}")

        dst_dir = backups_dir / "01"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in backup_files:
            try:
                shutil.copy2(str(src), str(dst_dir / src.name))
            except Exception as e:
                _logger.warning(f"Backup copy {src.name} failed: {e}")
        _logger.info(f"Backup created: {dst_dir}")

    def _restore_from_backup(self, backup_dir: Path) -> None:
        for f in backup_dir.iterdir():
            if f.is_file():
                target = self.model_dir / f.name
                try:
                    shutil.copy2(str(f), str(target))
                except Exception as e:
                    _logger.warning(f"Restore {f.name} failed: {e}")
        _logger.info(f"Restored from backup: {backup_dir}")
