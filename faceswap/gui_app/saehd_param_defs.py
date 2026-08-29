from dataclasses import dataclass, field

from faceswap.gui_app.param_defs import ParamDef, ParamGroup, ParamType

SAEHD_PARAM_DEFS: list[ParamDef] = [
    ParamDef(
        key="resolution", label="分辨率:", type=ParamType.INT, default=256,
        min_val=64, max_val=640, step=16, align_multiple=16, group=ParamGroup.BASIC,
        tooltip="训练分辨率(像素)。必须是16的倍数(含'd'架构需32的倍数)。\n"
                "代码层面影响: res>=256时启用双尺度DSSIM(5+5), res<256时单尺度(10)\n"
                "  DSSIM filter_size = res/11.6 和 res/23.2\n"
                "  blur_sigma = res//32, style_blur_radius = res//8\n"
                "128: 快速迭代, 适合前期调试\n"
                "192/256: 平衡质量与速度(推荐)\n"
                "384/512: 高细节, 显存占用平方级增长\n"
                "人脸类型会自动设置默认值(wf=256,head=384)",
    ),
    ParamDef(
        key="face_type", label="人脸类型:", type=ParamType.STR, default="wf",
        choices=["wf", "head"], group=ParamGroup.BASIC,
        render_hint="new_row",
        tooltip="人脸裁剪类型(同时决定默认分辨率):\n"
                "wf = 宽全脸(256, 最通用, 推荐)\n"
                "head = 整个头部(384, 含头发, 适合发际线融合但难度大)",
    ),
    ParamDef(
        key="batch_size", label="批次大小:", type=ParamType.INT, default=8,
        min_val=1, max_val=32, group=ParamGroup.BASIC,
        tooltip="每次迭代处理的图像对数量。\n"
                "越大梯度越稳定，但显存占用越大\n"
                "RTX 3050 6GB: 建议2~4\n"
                "过小(1~2)可能导致训练震荡",
    ),
    ParamDef(
        key="archi", label="ae架构:", type=ParamType.STR, default="df",
        group=ParamGroup.BASIC, render_hint="archi",
        tooltip="模型架构:\n"
                "df = DeepFake(双解码器, 推荐)\n"
                "liae = 共享解码器(Inter_AB+Inter_B)\n"
                "后缀选项(可多选):\n"
                "  c = cos激活(x*cos(x)替代leaky_relu)\n"
                "  d = 分辨率倍增(pixel_shuffle, 需32倍数)\n"
                "  t = 深层(更多downscale+resblock)\n"
                "  u = 像素归一化(pixel_norm)",
    ),
    ParamDef(
        key="adabelief", label="优化器:", type=ParamType.STR, default="adabelief",
        choices=["adabelief", "rmsprop"], group=ParamGroup.BASIC,
        tooltip="优化器选择(均对齐DFL leras实现, 均支持lr_dropout固定mask/lr_cos/clipnorm):\n"
                "adabelief = AdaBelief(DFL默认, 推荐)\n"
                "  v_t=β₂v+(1-β₂)(g-m_t)², 有动量, 无bias correction\n"
                "  对AE重建任务的大梯度噪声更鲁棒\n"
                "rmsprop = RMSprop(DFL备选)\n"
                "  a_t=ρa+(1-ρ)g², 无动量, 建议配合lr_dropout=y\n"
                "注意: 不提供Adam/AdamW, 因其bias correction和weight decay\n"
                "与DFL的AE过拟合训练目标矛盾, 无理论或实验优势",
    ),
    ParamDef(
        key="lr", label="学习率:", type=ParamType.FLOAT, default=5e-5,
        min_val=1e-6, max_val=1e-2, step=1e-5, decimals=7, group=ParamGroup.BASIC,
        tooltip="初始学习率。各优化器推荐值:\n"
                "  AdaBelief: 5e-5~1e-4 (默认5e-5)\n"
                "  RMSprop: 1e-4~5e-4\n"
                "学习率过高→模型走捷径(不分离身份/属性), 换脸无效\n"
                "训练后期可降低学习率帮助收敛",
    ),
    ParamDef(
        key="lr_dropout", label="学习率衰减:", type=ParamType.STR, default="n",
        choices=["n", "y"], group=ParamGroup.BASIC,
        tooltip="学习率随机丢弃策略(对SRC和DST对称生效, 对齐DFL lr_dropout):\n"
                "n = 不启用(推荐初期)\n"
                "y = 启用, 30%概率保留参数更新, 70%概率跳过\n"
                "训练后期开启可帮助收敛到更精细的结果\n"
                "同时自动启用余弦退火(lr_cos=500)",
    ),
    ParamDef(
        key="random_warp", label="随机变形", type=ParamType.BOOL, default=True,
        group=ParamGroup.BASIC, render_hint="new_row",
        tooltip="对训练图像施加随机仿射变形。\n"
                "必须开启! 这是防止过拟合和提升泛化能力的关键\n"
                "仅在预训练或最终微调阶段可考虑关闭",
    ),
    ParamDef(
        key="random_src_flip", label="源随机翻转", type=ParamType.BOOL, default=True,
        group=ParamGroup.BASIC,
        tooltip="对源(src)图像随机水平翻转，增强数据多样性。",
    ),
    ParamDef(
        key="random_dst_flip", label="目标随机翻转", type=ParamType.BOOL, default=True,
        group=ParamGroup.BASIC,
        tooltip="对目标(dst)图像随机水平翻转，增强数据多样性。",
    ),
    ParamDef(
        key="random_hsv_power", label="随机色调偏移:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=1.0, step=0.01, decimals=3, group=ParamGroup.BASIC,
        tooltip="随机色调/饱和度/亮度偏移强度(仅对SRC生效, DST不受影响)。\n"
                "0 = 不增强(默认)\n"
                "推荐: 0.0~0.05, 过大会导致色彩失真\n"
                "适用场景: SRC与DST肤色/光照差异大时增强颜色鲁棒性\n"
                "不适用: 数据颜色已匹配或已开ct_mode时(叠加会 destabilize 训练目标)",
    ),
    ParamDef(
        key="ct_mode", label="颜色迁移:", type=ParamType.STR, default="none",
        choices=["none", "rct", "rct-p", "lct", "mkl", "idt", "sot", "lab-match"],
        group=ParamGroup.BASIC,
        tooltip="训练时颜色迁移(仅对SRC生效, 将SRC颜色迁移到DST色彩空间)。\n"
                "none = 不迁移\n"
                "lab-match = [推荐]色度匹配(L通道30%迁移, AB通道协方差匹配)\n"
                "rct = Reinhard(快速, LAB均值/标准差匹配)\n"
                "rct-p = 部分Reinhard(自适应混合, 更温和)\n"
                "lct = 线性迁移(LAB空间PCA匹配协方差)\n"
                "mkl = MKL迁移(最优传输线性近似, 效果好)\n"
                "idt = IDT迭代迁移(最精确但最慢)\n"
                "sot = 切片最优传输(质量好但慢)\n"
                "注意: 每张图片做不同迁移可能导致训练目标不稳定, 减慢收敛",
    ),
    ParamDef(
        key="clipgrad", label="梯度裁剪", type=ParamType.BOOL, default=False,
        group=ParamGroup.BASIC,
        tooltip="启用梯度全局L2范数裁剪(上限1.0)，防止梯度爆炸。\n对齐DFL clipnorm=1.0",
    ),
    ParamDef(
        key="pretrain", label="预训练模式", type=ParamType.BOOL, default=False,
        group=ParamGroup.BASIC,
        tooltip="使用预训练数据(大量不同人脸)进行预训练。\n"
                "预训练数据目录: workspace/model/pretrain_faces/\n"
                "预训练后模型有更好的初始化, 可加速后续特定人物训练\n"
                "开启时自动: 关闭face_style/bg_style/GAN/true_face损失, 开启uniform_yaw",
    ),
    ParamDef(
        key="amp_mode", label="混合精度:", type=ParamType.STR, default="fp16",
        choices=["fp32", "fp16", "bf16"],
        group=ParamGroup.BASIC,
        tooltip="混合精度训练模式:\n"
                "fp32: 全精度, 最稳定但最慢\n"
                "fp16: 半精度+GradScaler, 省显存加速(推荐)\n"
                "bf16: BF16半精度, 精度不足易导致梯度爆炸, 不推荐\n"
                "推荐: fp16 > fp32",
    ),
    ParamDef(
        key="target_iter", label="目标迭代数:", type=ParamType.INT, default=0,
        min_val=0, max_val=9999999, step=10000, group=ParamGroup.BASIC,
        tooltip="训练目标迭代次数。\n"
                "0 = 无限训练(手动停止)\n"
                "推荐: 100000~500000(视数据量而定)",
    ),
    ParamDef(
        key="backup_interval", label="保存间隔:", type=ParamType.INT, default=0,
        min_val=0, max_val=1000000, step=1000, group=ParamGroup.BASIC,
        tooltip="每隔多少次迭代自动保存模型。\n"
                "0 = 不自动保存(仅训练结束时保存)\n"
                "推荐: 10000~50000",
    ),

    ParamDef(
        key="ae_dims", label="瓶颈维度:", type=ParamType.INT, default=256,
        min_val=32, max_val=1024, step=32, group=ParamGroup.BASIC,
        tooltip="Inter中间层瓶颈通道维度, 决定模型容量。\n"
                "256 = 标准值(推荐)\n"
                "384/512 = 保留更多细节, 但易过拟合且训练慢",
    ),
    ParamDef(
        key="e_dims", label="编码器维度:", type=ParamType.INT, default=64,
        min_val=16, max_val=256, step=2, group=ParamGroup.BASIC,
        tooltip="编码器基础通道维度, 控制特征提取能力。\n"
                "64 = 标准值(推荐)\n"
                "96/128 = 更强表达能力, 但计算量增加",
    ),
    ParamDef(
        key="d_dims", label="解码器维度:", type=ParamType.INT, default=64,
        min_val=16, max_val=256, step=2, group=ParamGroup.BASIC,
        tooltip="解码器基础通道维度, 控制生成细节丰富度。\n"
                "64 = 标准值(推荐)\n"
                "96/128 = 更精细细节, 但计算量增加",
    ),
    ParamDef(
        key="d_mask_dims", label="遮罩解码器维度:", type=ParamType.INT, default=22,
        min_val=8, max_val=128, step=2, group=ParamGroup.BASIC,
        render_hint="new_row",
        tooltip="遮罩解码器基础通道维度, 控制遮罩精细度。\n通常小于d_dims, 22是DFL默认值",
    ),

    ParamDef(
        key="masked_training", label="遮罩训练", type=ParamType.BOOL, default=True,
        group=ParamGroup.FACE_DETAIL,
        tooltip="仅在人脸遮罩区域内计算损失(对SRC和DST对称生效)。\n"
                "推荐开启, 避免背景区域干扰训练",
    ),
    ParamDef(
        key="eyes_mouth_prio", label="眼嘴优先", type=ParamType.BOOL, default=False,
        group=ParamGroup.FACE_DETAIL,
        tooltip="对眼睛和嘴巴区域施加300倍L1损失(对SRC和DST对称生效)。\n"
                "人眼对这些区域最敏感, 开启可显著提升观感\n"
                "训练中后期开启效果更佳",
    ),
    ParamDef(
        key="uniform_yaw", label="均匀偏航采样", type=ParamType.BOOL, default=False,
        group=ParamGroup.FACE_DETAIL,
        tooltip="均匀采样不同偏航角(左右转头)的人脸。\n"
                "数据中正脸过多侧脸过少时开启\n"
                "可显著改善侧脸换脸效果",
    ),
    ParamDef(
        key="blur_out_mask", label="模糊遮罩外区域", type=ParamType.BOOL, default=False,
        group=ParamGroup.FACE_DETAIL,
        tooltip="对遮罩外区域施加高斯模糊(对SRC和DST对称生效, 对齐DFL blur_out_mask)。\n"
                "减少遮罩边缘的突变, 改善背景过渡自然度\n"
                "预训练时建议关闭",
    ),
    ParamDef(
        key="multiscale_loss_power", label="多尺度损失强度:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=10.0, step=0.5, decimals=2, group=ParamGroup.FACE_DETAIL,
        tooltip="多尺度重建损失强度(对SRC和DST对称生效, 非DFL标准, 自研增强)。\n"
                "0 = 不启用(默认, 对齐DFL)\n"
                "推荐: 1.0~3.0\n"
                "对预测和目标下采样到64x64计算额外MSE损失,\n"
                "关注脸型轮廓/肤色等低频信息而非细节,\n"
                "对大角度侧脸和模糊脸特别有效:\n"
                "  全分辨率损失: 被遮挡区域产生无意义梯度\n"
                "  低分辨率损失: 脸型轮廓仍可见, 梯度有效",
    ),
    ParamDef(
        key="visibility_loss_power", label="可见性损失强度:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=10.0, step=0.5, decimals=2, group=ParamGroup.FACE_DETAIL,
        tooltip="可见性加权损失强度(对SRC和DST对称生效, 非DFL标准, 自研增强)。\n"
                "0 = 不启用(默认, 对齐DFL)\n"
                "推荐: 1.0~5.0\n"
                "利用元数据中landmarks_106_visibility生成可见性mask,\n"
                "可见区域梯度正常, 遮挡区域梯度被抑制。\n"
                "正脸(全可见)无效果, 大角度侧脸自动抑制遮挡区域。\n"
                "注: 仅对人工标注了可见性的数据生效,\n"
                "insightface自动提取的数据全可见(无效果但安全)",
    ),

    ParamDef(
        key="true_face_power", label="真脸判别强度:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=10.0, step=0.1, decimals=3, group=ParamGroup.LOSS_SAMPLING,
        archi_filter=["df"],
        tooltip="真脸判别器强度(仅df架构, 仅对SRC生效)。\n"
                "0 = 不启用(默认)\n"
                "0.01~0.1 = 将SRC的latent code推向DST的code分布\n"
                "效果: 换脸结果整体(五官+轮廓)趋向DST → 'DST五官+DST轮廓'\n"
                "与face_style_power效果相反, 实际使用时二选一, 不要同时开\n"
                "对齐DFL true_face_power",
    ),
    ParamDef(
        key="face_style_power", label="人脸风格强度:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=100.0, step=1.0, decimals=2, group=ParamGroup.LOSS_SAMPLING,
        tooltip="人脸风格损失强度(仅对SRC生效, 基于Gram矩阵style_loss)。\n"
                "0 = 不启用(默认)\n"
                "推荐: 1~100, 实际权重=10000×(power/100)\n"
                "效果: 匹配纹理/风格但不改空间结构 → 'SRC五官+DST轮廓'\n"
                "与true_face_power效果相反, 实际使用时二选一, 不要同时开\n"
                "预训练时自动关闭\n"
                "对齐DFL face_style_power",
    ),
    ParamDef(
        key="bg_style_power", label="背景风格强度:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=100.0, step=1.0, decimals=2, group=ParamGroup.LOSS_SAMPLING,
        tooltip="背景区域风格损失强度(仅对SRC生效, 遮罩外区域的DSSIM+MSE)。\n"
                "0 = 不启用(默认)\n"
                "推荐: 1~100, 改善背景区域自然度\n"
                "预训练时自动关闭\n"
                "对齐DFL bg_style_power",
    ),
    ParamDef(
        key="gan_power", label="GAN强度:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.LOSS_SAMPLING,
        tooltip="GAN对抗损失强度(仅对SRC重建生效, DST不参与GAN)。\n"
                "0 = 不启用(推荐初期)\n"
                "待重建稳定后(如50k iter)开启至0.1~1.0\n"
                "过高会导致伪影, 典型值0.1\n"
                "同时自动添加TV正则(1e-6)和anti_mask MSE(0.02)仅对SRC\n"
                "对齐DFL gan_power",
    ),
    ParamDef(
        key="gan_patch_size", label="GAN块大小:", type=ParamType.INT, default=16,
        min_val=3, max_val=640, step=1, group=ParamGroup.LOSS_SAMPLING,
        tooltip="GAN判别器的Patch大小(find_archi自动匹配最接近的感受野)。\n"
                "推荐: 16~32\n"
                "越大判别器感受野越大",
    ),
    ParamDef(
        key="gan_dims", label="GAN维度:", type=ParamType.INT, default=16,
        min_val=4, max_val=64, step=1, group=ParamGroup.LOSS_SAMPLING,
        tooltip="GAN判别器的基础通道维度。\n"
                "推荐: 16~64\n"
                "越大判别器容量越大",
    ),

    ParamDef(
        key="enable_torch_compile", label="torch.compile加速", type=ParamType.BOOL, default=False,
        group=ParamGroup.OPTIMIZATION,
        tooltip="启用torch.compile编译加速(PyTorch≥2.0)。\n"
                "默认关闭: 6GB显存下aot_eager后端只做AOT编译不做算子融合,\n"
                "无实际加速反而增加编译时间和显存开销。\n"
                "8GB+显存可用inductor后端获得真正加速(需PYTHONUTF8=1)。\n"
                "首次迭代触发编译(约30~120秒), 编译失败自动降级为eager模式",
    ),
]


@dataclass
class SubGroupDef:
    subgroup_id: int
    subgroup_name: str
    param_keys: list
    group_basis: str
    full_row_keys: list = field(default_factory=list)


SAEHD_SUBGROUPS: list = [
    SubGroupDef(
        subgroup_id=1, subgroup_name="模型架构",
        param_keys=["archi", "ae_dims", "e_dims", "d_dims",
                     "d_mask_dims", "face_type", "resolution", "batch_size"],
        group_basis="模型结构、通道维度与训练规格",
    ),
    SubGroupDef(
        subgroup_id=2, subgroup_name="优化器与学习率",
        param_keys=["adabelief", "lr", "lr_dropout"],
        group_basis="优化器类型与学习率策略",
    ),
    SubGroupDef(
        subgroup_id=3, subgroup_name="数据增强",
        param_keys=["random_hsv_power", "ct_mode",
                     "random_warp", "random_src_flip", "random_dst_flip"],
        group_basis="训练数据增强与色彩处理",
    ),
    SubGroupDef(
        subgroup_id=9, subgroup_name="人脸细节",
        param_keys=["masked_training", "eyes_mouth_prio", "uniform_yaw", "blur_out_mask",
                     "multiscale_loss_power", "visibility_loss_power"],
        group_basis="人脸区域采样与细节处理策略",
    ),
    SubGroupDef(
        subgroup_id=10, subgroup_name="损失函数",
        param_keys=["true_face_power", "face_style_power", "bg_style_power"],
        group_basis="DFL标准损失项权重, 0=不启用",
    ),
    SubGroupDef(
        subgroup_id=11, subgroup_name="GAN对抗",
        param_keys=["gan_power", "gan_patch_size", "gan_dims"],
        group_basis="GAN对抗训练, 仅gan_power>0时生效",
    ),
    SubGroupDef(
        subgroup_id=4, subgroup_name="训练控制",
        param_keys=["target_iter", "backup_interval", "pretrain"],
        group_basis="训练过程控制参数",
    ),
    SubGroupDef(
        subgroup_id=5, subgroup_name="数值稳定性与显存",
        param_keys=["amp_mode", "clipgrad", "enable_torch_compile"],
        group_basis="数值稳定性、显存占用、训练加速",
    ),
]


def get_saehd_params_by_group(group: ParamGroup) -> list[ParamDef]:
    return [p for p in SAEHD_PARAM_DEFS if p.group == group]


def get_saehd_params_by_keys(keys: list) -> list[ParamDef]:
    key_to_param = {p.key: p for p in SAEHD_PARAM_DEFS}
    result = []
    for k in keys:
        p = key_to_param.get(k)
        if p is not None:
            result.append(p)
    return result
