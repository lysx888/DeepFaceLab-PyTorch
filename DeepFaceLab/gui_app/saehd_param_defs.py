from DeepFaceLab.gui_app.param_defs import ParamDef, ParamGroup, ParamType

SAEHD_PARAM_DEFS: list[ParamDef] = [
    # === BASIC — 分辨率/FaceType/训练基础 (DFL顺序) ===
    ParamDef(key="resolution", label="Resolution:", type=ParamType.INT, default=128, min_val=64, max_val=640, step=16, align_multiple=16, group=ParamGroup.BASIC),
    ParamDef(key="face_type", label="Face type:", type=ParamType.STR, default="whole_face", choices=["half", "full", "whole_face", "head", "mid_full"], group=ParamGroup.BASIC),
    ParamDef(key="batch_size", label="Batch size:", type=ParamType.INT, default=0, min_val=0, max_val=64, group=ParamGroup.BASIC),
    ParamDef(key="learning_rate", label="Learning rate:", type=ParamType.FLOAT, default=5e-5, min_val=1e-6, max_val=1e-2, step=1e-5, decimals=6, group=ParamGroup.BASIC),
    ParamDef(key="optimizer", label="Optimizer:", type=ParamType.STR, default="adabelief", choices=["adabelief", "adam"], group=ParamGroup.BASIC),
    ParamDef(key="random_warp", label="Random warp", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="random_flip", label="Random flip", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="random_hsv_power", label="Random hue/sat/light:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=0.3, step=0.01, decimals=3, group=ParamGroup.BASIC),
    ParamDef(key="lr_dropout", label="LR dropout:", type=ParamType.STR, default="n", choices=["n", "y", "cpu"], group=ParamGroup.BASIC),
    ParamDef(key="pretrain", label="Pretrain mode", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    ParamDef(key="random_ct", label="Random color transfer", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    ParamDef(key="ct_mode", label="Color transfer mode:", type=ParamType.STR, default="none", choices=["none", "rct", "mkl", "lct", "idt", "sot-m"], group=ParamGroup.BASIC),
    ParamDef(key="random_ct_sample_size", label="CT sample size:", type=ParamType.INT, default=100, min_val=10, max_val=500, group=ParamGroup.BASIC),
    ParamDef(key="ca_weights", label="CA weight init", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    ParamDef(key="target_iter", label="Target iterations:", type=ParamType.INT, default=0, min_val=0, max_val=9999999, step=10000, group=ParamGroup.BASIC),
    ParamDef(key="autobackup_hour", label="Autobackup (hours):", type=ParamType.INT, default=0, min_val=0, max_val=24, group=ParamGroup.BASIC),
    ParamDef(key="write_preview_history", label="Save preview history", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    # === 扩展参数 (非DFL标准) ===
    ParamDef(key="use_amp", label="混合精度 [扩展]", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="gradient_clip", label="梯度裁剪 [扩展]", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    ParamDef(key="save_interval_min", label="保存间隔(分) [扩展]:", type=ParamType.INT, default=15, min_val=1, max_val=120, group=ParamGroup.BASIC),
    ParamDef(key="preview_interval_sec", label="预览间隔(秒) [扩展]:", type=ParamType.INT, default=60, min_val=10, max_val=600, group=ParamGroup.BASIC),

    # === ARCHITECTURE ===
    ParamDef(key="architecture", label="AE architecture:", type=ParamType.STR, default="liae-ud", choices=["df", "liae", "df-d", "liae-d", "df-ud", "liae-ud", "df-udt", "liae-udt", "df-t", "liae-t", "df-td", "liae-td"], group=ParamGroup.ARCHITECTURE),
    ParamDef(key="ae_dims", label="AutoEncoder dimensions:", type=ParamType.INT, default=256, min_val=32, max_val=1024, step=32, decimals=0, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="e_dims", label="Encoder dimensions:", type=ParamType.INT, default=64, min_val=16, max_val=256, step=16, decimals=0, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="d_dims", label="Decoder dimensions:", type=ParamType.INT, default=64, min_val=16, max_val=256, step=16, decimals=0, group=ParamGroup.ARCHITECTURE),

    # === FACE_DETAIL ===
    ParamDef(key="learn_mask", label="Learn mask", type=ParamType.BOOL, default=True, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="eyes_mouth_prio", label="Eyes and mouth priority", type=ParamType.BOOL, default=False, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="masked_training", label="Masked training", type=ParamType.BOOL, default=True, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="blur_out_mask", label="Blur out mask", type=ParamType.BOOL, default=False, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="src_face_scale", label="Src face scale %:", type=ParamType.INT, default=0, min_val=-30, max_val=30, group=ParamGroup.FACE_DETAIL),

    # === LOSS_SAMPLING ===
    ParamDef(key="face_style_power", label="Face style power:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=100.0, step=1.0, decimals=1, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="bg_style_power", label="BG style power:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=100.0, step=1.0, decimals=1, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="true_face_power", label="True face power:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=1.0, step=0.01, decimals=4, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="pixel_loss", label="Pixel loss (after 20k)", type=ParamType.BOOL, default=False, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="gan_power", label="GAN power:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="gan_dims", label="GAN dimensions:", type=ParamType.INT, default=16, min_val=8, max_val=64, step=8, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="gan_patch_size", label="GAN patch size:", type=ParamType.INT, default=0, min_val=0, max_val=128, step=8, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="uniform_yaw", label="Uniform yaw sampling", type=ParamType.BOOL, default=False, group=ParamGroup.LOSS_SAMPLING),
]


def get_saehd_params_by_group(group: ParamGroup) -> list[ParamDef]:
    return [p for p in SAEHD_PARAM_DEFS if p.group == group]
