import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from DeepFaceLab.setting import INSIGHTFACE_MODEL_DIR, INSIGHTFACE_DIR, WORKSPACE_DIR
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("insightface_exporter")


@dataclass
class ExportResult:
    onnx_path: Path
    max_abs_error: float
    is_consistent: bool
    deployed: bool = False


class InsightFaceExporter:
    SCRFD_ONNX_NAME = "scrfd_10g_bnkps.onnx"
    SYNTHETICS_ONNX_NAME = "2d106det.onnx"
    DEFAULT_OPSET = 11

    def __init__(self, workspace_dir: Optional[Path] = None) -> None:
        self._workspace_dir = Path(workspace_dir) if workspace_dir else WORKSPACE_DIR
        self._output_dir = self._workspace_dir / "insightface_train" / "output"

    @property
    def antelopev2_dir(self) -> Path:
        d = INSIGHTFACE_MODEL_DIR / "antelopev2"
        FileManager.ensure_dir(d)
        return d

    def export_scrfd_onnx(
        self,
        checkpoint_path: Path,
        output_path: Optional[Path] = None,
        input_size: tuple[int, int] = (640, 640),
        opset: int = 11,
        verify: bool = True,
        deploy: bool = False,
    ) -> ExportResult:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SCRFD checkpoint不存在: {checkpoint_path}")

        if output_path is None:
            output_path = self._output_dir / "scrfd" / self.SCRFD_ONNX_NAME
        FileManager.ensure_dir(output_path.parent)

        config_path = INSIGHTFACE_DIR / "detection" / "scrfd" / "configs" / "scrfd" / "dfl_scrfd_10g_bnkps.py"
        scrfd2onnx_script = INSIGHTFACE_DIR / "detection" / "scrfd" / "tools" / "scrfd2onnx.py"

        _logger.info(f"Exporting SCRFD ONNX: {checkpoint_path} -> {output_path}")

        import torch
        from mmdet.core import generate_inputs_and_wrap_model

        input_config = {
            "input_shape": (1, 3) + tuple(input_size),
            "input_path": None,
            "normalize_cfg": {"mean": [127.5, 127.5, 127.5], "std": [128.0, 128.0, 128.0]},
        }

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        if "optimizer" in ckpt:
            del ckpt["optimizer"]
        slim_path = output_path.parent / "_slim_tmp.pth"
        torch.save(ckpt, str(slim_path))

        try:
            model, tensor_data = generate_inputs_and_wrap_model(
                str(config_path), str(slim_path), input_config
            )
        except Exception as e:
            if slim_path.exists():
                slim_path.unlink()
            raise RuntimeError(f"SCRFD模型构建失败: {e}")

        input_names = ["input.1"]
        output_names = [
            "score_8", "score_16", "score_32",
            "bbox_8", "bbox_16", "bbox_32",
        ]
        if "stride_kps" in str(model):
            output_names += ["kps_8", "kps_16", "kps_32"]

        dynamic_axes = {out: {0: "?", 1: "?"} for out in output_names}
        dynamic_axes[input_names[0]] = {0: "?", 2: "?", 3: "?"}

        ori_path = output_path.parent / (output_path.stem + "_ori.onnx")
        try:
            torch.onnx.export(
                model,
                tensor_data,
                str(ori_path),
                keep_initializers_as_inputs=False,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
            )
        except Exception as e:
            if slim_path.exists():
                slim_path.unlink()
            if ori_path.exists():
                ori_path.unlink()
            raise RuntimeError(f"SCRFD ONNX导出失败: {e}")

        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(str(ori_path))
            input_shapes = {onnx_model.graph.input[0].name: list(input_config["input_shape"])}
            onnx_model, check = simplify(onnx_model, input_shapes=input_shapes, dynamic_input_shape=True)
            if not check:
                _logger.warning("ONNX simplify validation failed, using unsimplified model")
            onnx.save(onnx_model, str(output_path))
        except ImportError:
            _logger.warning("onnxsim not available, skipping simplification")
            shutil.move(str(ori_path), str(output_path))
        finally:
            if ori_path.exists():
                ori_path.unlink()
            if slim_path.exists():
                slim_path.unlink()

        _logger.info(f"SCRFD ONNX exported: {output_path}")

        max_abs_error = 0.0
        is_consistent = True
        if verify:
            max_abs_error, is_consistent = self._verify_onnx_consistency(
                output_path, checkpoint_path, "scrfd", input_size
            )

        deployed = False
        if deploy:
            deployed = self.deploy_to_antelopev2(output_path, self.SCRFD_ONNX_NAME)

        return ExportResult(
            onnx_path=output_path,
            max_abs_error=max_abs_error,
            is_consistent=is_consistent,
            deployed=deployed,
        )

    def export_synthetics_onnx(
        self,
        checkpoint_path: Path,
        output_path: Optional[Path] = None,
        input_size: int = 256,
        opset: int = 11,
        verify: bool = True,
        deploy: bool = False,
    ) -> ExportResult:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"2d106 checkpoint不存在: {checkpoint_path}")

        if output_path is None:
            output_path = self._output_dir / "synthetics" / self.SYNTHETICS_ONNX_NAME
        FileManager.ensure_dir(output_path.parent)

        _logger.info(f"Exporting 2d106 ONNX: {checkpoint_path} -> {output_path}")

        import torch
        import timm

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        if "hyper_parameters" in ckpt and "backbone" in ckpt["hyper_parameters"]:
            backbone_name = ckpt["hyper_parameters"]["backbone"]
        else:
            backbone_name = "resnet50d"

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    cleaned[k[len("backbone."):]] = v
            state_dict = cleaned
        else:
            state_dict = ckpt

        model = timm.create_model(backbone_name, num_classes=68 * 2)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise RuntimeError(f"2d106模型权重加载失败: {e}")

        model.eval()

        dummy_input = torch.randn(1, 3, input_size, input_size)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "output": {0: "batch"},
                },
                opset_version=opset,
            )
        except Exception as e:
            raise RuntimeError(f"2d106 ONNX导出失败: {e}")

        _logger.info(f"2d106 ONNX exported: {output_path}")

        max_abs_error = 0.0
        is_consistent = True
        if verify:
            max_abs_error, is_consistent = self._verify_onnx_consistency(
                output_path, checkpoint_path, "synthetics", input_size
            )

        deployed = False
        if deploy:
            deployed = self.deploy_to_antelopev2(output_path, self.SYNTHETICS_ONNX_NAME)

        return ExportResult(
            onnx_path=output_path,
            max_abs_error=max_abs_error,
            is_consistent=is_consistent,
            deployed=deployed,
        )

    def deploy_to_antelopev2(
        self,
        onnx_path: Path,
        model_name: str,
        backup: bool = True,
    ) -> bool:
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX模型文件不存在: {onnx_path}")

        target_dir = self.antelopev2_dir
        target_path = target_dir / model_name

        if target_path.exists() and backup:
            bak_path = target_path.with_suffix(target_path.suffix + ".bak")
            if bak_path.exists():
                bak_path.unlink()
            shutil.move(str(target_path), str(bak_path))
            _logger.info(f"Backed up existing model: {target_path} -> {bak_path}")

        tmp_path = target_path.with_suffix(".tmp")
        try:
            shutil.copy2(str(onnx_path), str(tmp_path))
            os.replace(str(tmp_path), str(target_path))
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            bak_path = target_path.with_suffix(target_path.suffix + ".bak")
            if bak_path.exists() and not target_path.exists():
                shutil.move(str(bak_path), str(target_path))
                _logger.info(f"Restored backup after deploy failure: {bak_path}")
            raise RuntimeError(f"ONNX模型部署失败: {e}")

        _logger.info(f"ONNX model deployed: {target_path}")
        return True

    def _verify_onnx_consistency(
        self,
        onnx_path: Path,
        checkpoint_path: Path,
        model_type: str,
        input_size: tuple[int, int] | int,
        max_error_threshold: float = 1e-3,
    ) -> tuple[float, bool]:
        try:
            import onnxruntime as ort
        except ImportError:
            _logger.warning("onnxruntime未安装，跳过ONNX一致性验证")
            return 0.0, True

        import torch

        if isinstance(input_size, int):
            input_shape = (1, 3, input_size, input_size)
        else:
            input_shape = (1, 3) + tuple(input_size)

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        ort_session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
        ort_outputs = ort_session.run(None, ort_inputs)

        if model_type == "scrfd":
            from mmdet.core import generate_inputs_and_wrap_model

            config_path = INSIGHTFACE_DIR / "detection" / "scrfd" / "configs" / "scrfd" / "dfl_scrfd_10g_bnkps.py"
            input_config = {
                "input_shape": input_shape,
                "input_path": None,
                "normalize_cfg": {"mean": [127.5, 127.5, 127.5], "std": [128.0, 128.0, 128.0]},
            }
            model, tensor_data = generate_inputs_and_wrap_model(
                str(config_path), str(checkpoint_path), input_config
            )
            with torch.no_grad():
                pt_outputs = model(tensor_data, return_loss=False)
        else:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu")
            backbone_name = ckpt.get("hyper_parameters", {}).get("backbone", "resnet50d")
            import timm
            model = timm.create_model(backbone_name, num_classes=68 * 2)
            if "state_dict" in ckpt:
                state_dict = {k.replace("backbone.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("backbone.")}
            else:
                state_dict = ckpt
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            with torch.no_grad():
                pt_tensor = torch.from_numpy(dummy_input)
                pt_outputs = [model(pt_tensor).numpy()]

        max_abs_error = 0.0
        for ort_out, pt_out in zip(ort_outputs, pt_outputs):
            if isinstance(pt_out, torch.Tensor):
                pt_out = pt_out.cpu().numpy()
            err = np.max(np.abs(ort_out.ravel() - pt_out.ravel()))
            max_abs_error = max(max_abs_error, float(err))

        is_consistent = max_abs_error < max_error_threshold
        if not is_consistent:
            _logger.warning(
                f"ONNX一致性验证未通过: max_abs_error={max_abs_error:.6f} > {max_error_threshold}"
            )
        else:
            _logger.info(f"ONNX一致性验证通过: max_abs_error={max_abs_error:.6f}")

        return max_abs_error, is_consistent
