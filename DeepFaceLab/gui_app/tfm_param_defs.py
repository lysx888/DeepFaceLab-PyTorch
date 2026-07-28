from DeepFaceLab.gui_app.param_defs import ParamDef, ParamGroup, ParamType
from DeepFaceLab.models.tfm_model import _TFM_PRESETS

TFM_PARAM_DEFS: list[ParamDef] = [
    # === BASIC ===
    ParamDef(key="resolution", label="分辨率:", type=ParamType.INT, default=128, min_val=64, max_val=256, step=16, align_multiple=16, group=ParamGroup.BASIC),
    ParamDef(key="face_type", label="人脸类型:", type=ParamType.STR, default="whole_face", choices=["half", "mid_full", "full", "whole_face", "head"], group=ParamGroup.BASIC),
    ParamDef(key="batch_size", label="批次大小:", type=ParamType.INT, default=4, min_val=1, max_val=64, group=ParamGroup.BASIC),
    ParamDef(key="learning_rate", label="学习率:", type=ParamType.FLOAT, default=1e-4, min_val=1e-6, max_val=1e-2, step=1e-5, decimals=6, group=ParamGroup.BASIC),
    ParamDef(key="use_amp", label="混合精度", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="random_warp", label="随机变形", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="gan_power", label="GAN强度:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.BASIC),
    ParamDef(key="random_hsv_power", label="HSV增强:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=0.3, step=0.01, decimals=3, group=ParamGroup.BASIC),
    ParamDef(key="lr_schedule", label="学习率调度:", type=ParamType.STR, default="constant", choices=["constant", "cosine_annealing"], group=ParamGroup.BASIC),
    ParamDef(key="save_interval_min", label="保存间隔(分):", type=ParamType.INT, default=15, min_val=1, max_val=120, group=ParamGroup.BASIC),
    ParamDef(key="preview_interval_sec", label="预览间隔(秒):", type=ParamType.INT, default=60, min_val=10, max_val=600, group=ParamGroup.BASIC),
    ParamDef(key="gradient_clip", label="梯度裁剪:", type=ParamType.FLOAT, default=1.0, min_val=0.0, max_val=10.0, step=0.1, decimals=2, group=ParamGroup.BASIC),
    ParamDef(key="random_flip", label="随机翻转", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="color_transfer", label="颜色迁移:", type=ParamType.STR, default="none", choices=["none", "rct", "mkl"], group=ParamGroup.BASIC),
    # === ARCHITECTURE ===
    ParamDef(key="model_preset", label="模型预设:", type=ParamType.STR, default="medium", choices=["tiny", "small", "medium", "large"], group=ParamGroup.ARCHITECTURE),
    ParamDef(key="window_size", label="窗口大小:", type=ParamType.STR, default="8", choices=["4", "8", "16"], group=ParamGroup.ARCHITECTURE),
    ParamDef(key="ae_dims", label="瓶颈维度:", type=ParamType.INT, default=256, min_val=64, max_val=512, step=32, decimals=0, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="gradient_checkpoint", label="梯度检查点", type=ParamType.BOOL, default=False, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="use_compile", label="torch.compile", type=ParamType.BOOL, default=False, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="embed_dim", label="嵌入维度:", type=ParamType.INT, default=96, min_val=16, max_val=256, group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    ParamDef(key="depths", label="块数:", type=ParamType.STR, default="[2,2,6,2]", group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    ParamDef(key="num_heads", label="头数:", type=ParamType.STR, default="[3,6,12,24]", group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    ParamDef(key="base_channels", label="基础通道:", type=ParamType.INT, default=512, min_val=32, max_val=1024, group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    ParamDef(key="w_dim", label="W+维度:", type=ParamType.INT, default=512, min_val=64, max_val=1024, group=ParamGroup.ARCHITECTURE, preset_controlled=True),
    # === FACE_DETAIL ===
    ParamDef(key="eye_priority", label="眼睛优先:", type=ParamType.FLOAT, default=1.0, min_val=0.5, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="mouth_priority", label="嘴巴优先:", type=ParamType.FLOAT, default=1.0, min_val=0.5, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="nose_priority", label="鼻子优先:", type=ParamType.FLOAT, default=1.0, min_val=0.5, max_val=3.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="jaw_priority", label="下颌优先:", type=ParamType.FLOAT, default=1.0, min_val=0.5, max_val=3.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="face_style_power", label="脸部风格:", type=ParamType.FLOAT, default=5.0, min_val=0.1, max_val=10.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="bg_style_power", label="背景风格:", type=ParamType.FLOAT, default=2.0, min_val=0.0, max_val=10.0, step=0.1, decimals=2, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="enable_mask", label="启用遮罩", type=ParamType.BOOL, default=True, group=ParamGroup.FACE_DETAIL),
    # === LOSS_SAMPLING ===
    ParamDef(key="perceptual_weight", label="自重建感知:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=1.0, step=0.01, decimals=3, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="uniform_yaw_sampling", label="均匀角度采样", type=ParamType.BOOL, default=False, group=ParamGroup.LOSS_SAMPLING),
]


def get_tfm_params_by_group(group: ParamGroup) -> list[ParamDef]:
    return [p for p in TFM_PARAM_DEFS if p.group == group]


class TFMPresetManager:
    def __init__(self, arch_group, preset_combo) -> None:
        self._arch_group = arch_group
        self._preset_combo = preset_combo
        self._preset_map = _TFM_PRESETS
        preset_combo.currentTextChanged.connect(self.on_preset_changed)

    def on_preset_changed(self, preset_name: str) -> None:
        if preset_name not in self._preset_map:
            return
        cfg = self._preset_map[preset_name]
        arch_values = {
            "embed_dim": cfg["embed_dim"],
            "depths": str(cfg["depths"]),
            "num_heads": str(cfg["num_heads"]),
            "base_channels": cfg["base_channels"],
            "w_dim": cfg["w_dim"],
        }
        self._arch_group.set_values(arch_values)
        self._arch_group.set_editable(False, keys=["embed_dim", "depths", "num_heads", "base_channels", "w_dim"])

    def apply_saved_preset(self, saved_config: dict) -> None:
        preset = saved_config.get("model_preset", "medium")
        idx = self._preset_combo.findText(preset)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        self.on_preset_changed(preset)

    def get_current_preset(self) -> str:
        return self._preset_combo.currentText()
