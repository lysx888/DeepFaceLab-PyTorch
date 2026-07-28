from DeepFaceLab.gui_app.param_defs import ParamDef, ParamGroup, ParamType

SAEHD_PARAM_DEFS: list[ParamDef] = [
    # === BASIC ===
    ParamDef(key="resolution", label="分辨率:", type=ParamType.INT, default=128, min_val=64, max_val=640, step=16, align_multiple=16, group=ParamGroup.BASIC),
    ParamDef(key="face_type", label="人脸类型:", type=ParamType.STR, default="whole_face", choices=["half", "mid_full", "full", "whole_face", "head"], group=ParamGroup.BASIC),
    ParamDef(key="batch_size", label="批次大小:", type=ParamType.INT, default=8, min_val=1, max_val=64, group=ParamGroup.BASIC),
    ParamDef(key="learning_rate", label="学习率:", type=ParamType.FLOAT, default=5e-5, min_val=1e-6, max_val=1e-2, step=1e-5, decimals=6, group=ParamGroup.BASIC),
    ParamDef(key="adabelief", label="AdaBelief优化器", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="use_amp", label="混合精度", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="random_warp", label="随机变形", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="random_hsv_power", label="HSV增强:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=0.3, step=0.01, decimals=3, group=ParamGroup.BASIC),
    ParamDef(key="random_flip", label="随机翻转", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="color_transfer", label="颜色迁移:", type=ParamType.STR, default="none", choices=["none", "rct", "mkl"], group=ParamGroup.BASIC),
    ParamDef(key="save_interval_min", label="保存间隔(分):", type=ParamType.INT, default=15, min_val=1, max_val=120, group=ParamGroup.BASIC),
    ParamDef(key="preview_interval_sec", label="预览间隔(秒):", type=ParamType.INT, default=60, min_val=10, max_val=600, group=ParamGroup.BASIC),
    ParamDef(key="lr_dropout", label="LR Dropout:", type=ParamType.STR, default="n", choices=["n", "y", "cpu"], group=ParamGroup.BASIC),
    ParamDef(key="pretrain", label="预训练模式", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    ParamDef(key="gradient_clip", label="梯度裁剪", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    ParamDef(key="gan_power", label="GAN强度:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.BASIC),
    ParamDef(key="gan_dims", label="GAN维度:", type=ParamType.INT, default=16, min_val=8, max_val=64, step=8, group=ParamGroup.BASIC),
    # === ARCHITECTURE ===
    ParamDef(key="architecture", label="架构:", type=ParamType.STR, default="df", choices=["df", "liae", "df-d", "liae-d", "df-ud", "liae-ud", "df-udt", "liae-udt"], group=ParamGroup.ARCHITECTURE),
    ParamDef(key="ae_dims", label="自编码维度:", type=ParamType.INT, default=256, min_val=32, max_val=1024, step=32, decimals=0, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="e_dims", label="编码器维度:", type=ParamType.INT, default=64, min_val=16, max_val=256, step=16, decimals=0, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="d_dims", label="解码器维度:", type=ParamType.INT, default=64, min_val=16, max_val=256, step=16, decimals=0, group=ParamGroup.ARCHITECTURE),
    # === FACE_DETAIL ===
    ParamDef(key="eyes_mouth_prio", label="眼嘴优先", type=ParamType.BOOL, default=False, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="masked_training", label="遮罩训练", type=ParamType.BOOL, default=True, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="true_face_power", label="真脸强度:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=1.0, step=0.01, decimals=4, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="blur_out_mask", label="遮罩外模糊", type=ParamType.BOOL, default=False, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="face_style_power", label="面部风格:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=100.0, step=1.0, decimals=1, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="bg_style_power", label="背景风格:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=100.0, step=1.0, decimals=1, group=ParamGroup.FACE_DETAIL),
    # === LOSS_SAMPLING ===
    ParamDef(key="uniform_yaw", label="均匀角度采样", type=ParamType.BOOL, default=False, group=ParamGroup.LOSS_SAMPLING),
]


def get_saehd_params_by_group(group: ParamGroup) -> list[ParamDef]:
    return [p for p in SAEHD_PARAM_DEFS if p.group == group]
