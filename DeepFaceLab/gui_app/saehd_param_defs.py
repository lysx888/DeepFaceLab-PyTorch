from DeepFaceLab.gui_app.param_defs import ParamDef, ParamGroup, ParamType

SAEHD_PARAM_DEFS: list[ParamDef] = [
    # === BASIC — DFL用户交互参数 ===
    ParamDef(key="resolution", label="Resolution:", type=ParamType.INT, default=128, min_val=64, max_val=640, step=16, align_multiple=16, group=ParamGroup.BASIC),
    ParamDef(key="face_type", label="Face type:", type=ParamType.STR, default="wf", choices=["h", "mf", "f", "wf", "head"], group=ParamGroup.BASIC),
    ParamDef(key="batch_size", label="Batch size:", type=ParamType.INT, default=0, min_val=0, max_val=64, group=ParamGroup.BASIC),
    ParamDef(key="optimizer", label="Optimizer:", type=ParamType.STR, default="adamw", choices=["adamw", "adabelief", "adam", "rmsprop"], group=ParamGroup.BASIC),
    ParamDef(key="random_warp", label="Random warp", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="random_src_flip", label="Random src flip", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    ParamDef(key="random_dst_flip", label="Random dst flip", type=ParamType.BOOL, default=True, group=ParamGroup.BASIC),
    ParamDef(key="random_hsv_power", label="Random hue/sat/light:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=0.3, step=0.01, decimals=3, group=ParamGroup.BASIC),
    ParamDef(key="lr_dropout", label="LR dropout:", type=ParamType.STR, default="n", choices=["n", "y", "cpu"], group=ParamGroup.BASIC),
    ParamDef(key="ct_mode", label="Color transfer:", type=ParamType.STR, default="none", choices=["none", "rct", "lct", "mkl", "idt", "sot"], group=ParamGroup.BASIC),
    ParamDef(key="clipgrad", label="Gradient clipping", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    ParamDef(key="pretrain", label="Pretrain mode", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),
    ParamDef(key="target_iter", label="Target iterations:", type=ParamType.INT, default=0, min_val=0, max_val=9999999, step=10000, group=ParamGroup.BASIC),
    ParamDef(key="autobackup_hour", label="Autobackup (hours):", type=ParamType.INT, default=0, min_val=0, max_val=24, group=ParamGroup.BASIC),
    ParamDef(key="write_preview_history", label="Save preview history", type=ParamType.BOOL, default=False, group=ParamGroup.BASIC),

    # === ARCHITECTURE — DFL首次运行参数 ===
    ParamDef(key="architecture", label="AE architecture:", type=ParamType.STR, default="df-ud", choices=["df", "liae", "df-d", "liae-d", "df-ud", "liae-ud", "df-udt", "liae-udt", "df-t", "liae-t", "df-td", "liae-td"], group=ParamGroup.ARCHITECTURE),
    ParamDef(key="ae_dims", label="AutoEncoder dims:", type=ParamType.INT, default=256, min_val=32, max_val=1024, step=32, decimals=0, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="e_dims", label="Encoder dims:", type=ParamType.INT, default=32, min_val=16, max_val=256, step=16, decimals=0, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="d_dims", label="Decoder dims:", type=ParamType.INT, default=32, min_val=16, max_val=256, step=16, decimals=0, group=ParamGroup.ARCHITECTURE),
    ParamDef(key="d_mask_dims", label="Decoder mask dims:", type=ParamType.INT, default=16, min_val=16, max_val=256, step=2, decimals=0, group=ParamGroup.ARCHITECTURE),

    # === FACE_DETAIL — DFL可覆盖参数 ===
    ParamDef(key="masked_training", label="Masked training", type=ParamType.BOOL, default=True, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="eyes_mouth_prio", label="Eyes and mouth priority", type=ParamType.BOOL, default=False, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="uniform_yaw", label="Uniform yaw sampling", type=ParamType.BOOL, default=False, group=ParamGroup.FACE_DETAIL),
    ParamDef(key="blur_out_mask", label="Blur out mask", type=ParamType.BOOL, default=False, group=ParamGroup.FACE_DETAIL),

    # === LOSS_SAMPLING — DFL可覆盖参数 ===
    ParamDef(key="face_style_power", label="Face style power:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=100.0, step=1.0, decimals=1, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="bg_style_power", label="BG style power:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=100.0, step=1.0, decimals=1, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="true_face_power", label="True face power:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=1.0, step=0.01, decimals=4, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="gan_power", label="GAN power:", type=ParamType.FLOAT, default=0.0, min_val=0.0, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="gan_patch_size", label="GAN patch size:", type=ParamType.INT, default=16, min_val=0, max_val=640, step=8, group=ParamGroup.LOSS_SAMPLING),
    ParamDef(key="gan_dims", label="GAN dimensions:", type=ParamType.INT, default=16, min_val=4, max_val=512, step=4, group=ParamGroup.LOSS_SAMPLING),
]

SAEHD_HIDDEN_PARAMS = {
    "learning_rate": 5e-5,
    "use_amp": True,
    "learn_mask": True,
    "random_ct_sample_size": 100,
    "ca_weights": False,
    "save_interval_min": 15,
    "preview_interval_sec": 60,
    "src_face_scale": 0,
    "pixel_loss": False,
}

DFL_FACE_TYPE_MAP = {
    "h": "half",
    "mf": "mid_full",
    "f": "full",
    "wf": "whole_face",
    "head": "head",
}


def get_saehd_params_by_group(group: ParamGroup) -> list[ParamDef]:
    return [p for p in SAEHD_PARAM_DEFS if p.group == group]
