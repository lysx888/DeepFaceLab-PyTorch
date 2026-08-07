from faceswap.gui_app.param_defs import ParamDef, ParamGroup, ParamType

SAEHD_PARAM_DEFS: list[ParamDef] = [
    ParamDef(
        key="resolution", label="分辨率:", type=ParamType.INT, default=256,
        min_val=64, max_val=640, step=16, align_multiple=16, group=ParamGroup.BASIC,
        tooltip="训练分辨率(像素)。必须是16的倍数(含'd'架构需32的倍数)。\n"
                "128: 快速迭代，适合前期调试\n"
                "192/256: 平衡质量与速度(推荐)\n"
                "384/512: 高细节，显存占用平方级增长\n"
                "人脸类型会自动设置默认值(wf=256,head=384)",
    ),
    ParamDef(
        key="face_type", label="人脸类型:", type=ParamType.STR, default="wf",
        choices=["h", "mf", "f", "wf", "head"], group=ParamGroup.BASIC,
        tooltip="人脸裁剪类型(同时决定默认分辨率):\n"
                "h = 半脸(64), mf = 中脸(128), f = 全脸(128)\n"
                "wf = 宽全脸(256, 最通用, 推荐)\n"
                "head = 整个头部(384, 含头发, 适合发际线融合但难度大)",
    ),
    ParamDef(
        key="batch_size", label="批次大小:", type=ParamType.INT, default=8,
        min_val=1, max_val=32, group=ParamGroup.BASIC,
        tooltip="每次迭代处理的图像对数量。\n"
                "越大梯度越稳定，但显存占用越大\n"
                "RTX 3090/4090 + BF16可尝试16~32\n"
                "过小(1~2)可能导致训练震荡",
    ),
    ParamDef(
        key="optimizer", label="优化器:", type=ParamType.STR, default="adam",
        choices=["adam", "adabelief", "adamw", "lion"], group=ParamGroup.BASIC,
        tooltip="训练优化器:\n"
                "adam = 标准Adam, 稳定可靠\n"
                "adabelief = 收敛更快更稳(推荐尝试)\n"
                "adamw = 解耦权重衰减(需设weight_decay=0)\n"
                "lion = 显存更省但可能不够稳定",
    ),
    ParamDef(
        key="lr", label="学习率:", type=ParamType.FLOAT, default=5e-5,
        min_val=1e-6, max_val=1e-2, step=1e-5, decimals=7, group=ParamGroup.BASIC,
        tooltip="初始学习率。各优化器推荐值:\n"
                "  Adam/AdamW: 1e-4~5e-4\n"
                "  AdaBelief: 5e-5~1e-4 (默认5e-5)\n"
                "  Lion: 1e-4~5e-4\n"
                "学习率过高→模型走捷径(不分离身份/属性), 换脸无效\n"
                "训练后期可降低学习率帮助收敛",
    ),
    ParamDef(
        key="lr_dropout", label="学习率衰减:", type=ParamType.STR, default="n",
        choices=["n", "y", "cpu"], group=ParamGroup.BASIC,
        tooltip="学习率随机丢弃策略:\n"
                "n = 不启用(推荐初期)\n"
                "y = 启用，随机将部分参数学习率置零\n"
                "cpu = 启用并在CPU上计算\n"
                "训练后期开启可帮助收敛到更精细的结果",
    ),
    ParamDef(
        key="lr_cos", label="余弦退火周期:", type=ParamType.INT, default=0,
        min_val=0, max_val=100000, step=100, group=ParamGroup.BASIC,
        tooltip="余弦退火学习率调度的周期迭代数。\n"
                "0 = 不使用\n"
                "设置后学习率按余弦曲线周期性变化\n"
                "有助于跳出局部最优, 推荐值: 50000~200000",
    ),
    ParamDef(
        key="random_warp", label="随机变形", type=ParamType.BOOL, default=True,
        group=ParamGroup.BASIC,
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
        tooltip="随机色调/饱和度/亮度偏移强度。\n"
                "0 = 不增强\n"
                "推荐: 0.0~0.05, 过大会导致色彩失真",
    ),
    ParamDef(
        key="ct_mode", label="颜色迁移:", type=ParamType.STR, default="none",
        choices=["none", "rct", "rct-p", "lct", "mkl", "idt", "sot", "hist-match", "lab-match"],
        group=ParamGroup.BASIC,
        tooltip="训练时的颜色迁移方法(将src颜色迁移到dst色彩空间):\n"
                "none = 不迁移\n"
                "rct = Reinhard(快速, LAB均值/标准差匹配)\n"
                "rct-p = 部分Reinhard(50%混合, 更温和)\n"
                "lct = 线性迁移(PCA匹配协方差)\n"
                "mkl = MKL迁移(最优传输线性近似, 效果好)\n"
                "idt = IDT迭代迁移(最精确但最慢)\n"
                "sot = 切片最优传输(质量好但慢)\n"
                "hist-match = 直方图匹配(处理多峰分布好)\n"
                "lab-match = 色度匹配(仅迁移AB通道, 保留亮度)\n"
                "DF架构建议开启以缓解色斑问题",
    ),
    ParamDef(
        key="clipgrad", label="梯度裁剪", type=ParamType.BOOL, default=False,
        group=ParamGroup.BASIC,
        tooltip="启用梯度裁剪(范数上限1.0)，防止梯度爆炸。\n训练不稳定/出现NaN时开启",
    ),
    ParamDef(
        key="pretrain", label="预训练模式", type=ParamType.BOOL, default=False,
        group=ParamGroup.BASIC,
        tooltip="使用预训练数据(大量不同人脸)进行预训练。\n"
                "预训练数据目录: workspace/model/pretrain_faces/\n"
                "预训练后模型有更好的初始化, 可加速后续特定人物训练\n"
                "开启时自动: 关闭GAN/变形/HSV/VGG, 开启uniform_yaw",
    ),
    ParamDef(
        key="amp_mode", label="混合精度:", type=ParamType.STR, default="bf16",
        choices=["fp32", "fp16", "bf16"],
        group=ParamGroup.BASIC,
        tooltip="混合精度训练模式:\n"
                "fp32: 全精度, 最稳定但最慢\n"
                "fp16: 半精度+GradScaler, 省显存加速\n"
                "bf16: BF16半精度, 需Ampere+GPU(RTX 30/40系列)\n"
                "推荐: bf16(有条件) > fp16 > fp32",
    ),
    ParamDef(
        key="gradient_checkpointing", label="梯度检查点", type=ParamType.BOOL, default=False,
        group=ParamGroup.BASIC,
        tooltip="启用梯度检查点, 用计算换显存。\n显存不足时开启, 会降低约20%训练速度",
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
        key="archi", label="AE架构:", type=ParamType.STR, default="liae-ud",
        choices=None, group=ParamGroup.ARCHITECTURE, render_hint="archi_buttons",
        tooltip="自编码器架构:\n"
                "DF = 双解码器, 身份保留强但易产生色斑\n"
                "LIAE = 共享解码器, 颜色一致性好(推荐)\n"
                "修饰符:\n"
                "  u = 像素归一化(推荐, 稳定训练)\n"
                "  d = 密集连接(推荐, 提升细节)\n"
                "  t = 更多下采样层(高分辨率时使用)\n"
                "  c = 余弦激活函数(替代ReLU)\n"
                "推荐新手: liae-ud",
    ),
    ParamDef(
        key="ae_dims", label="自编码器维度:", type=ParamType.INT, default=256,
        min_val=32, max_val=1024, step=32, group=ParamGroup.ARCHITECTURE,
        tooltip="自编码器瓶颈维度, 决定模型容量。\n"
                "256 = 标准值(推荐)\n"
                "384/512 = 保留更多细节, 但易过拟合且训练慢",
    ),
    ParamDef(
        key="e_dims", label="编码器维度:", type=ParamType.INT, default=64,
        min_val=16, max_val=256, step=2, group=ParamGroup.ARCHITECTURE,
        tooltip="编码器基础通道维度, 控制特征提取能力。\n"
                "64 = 标准值(推荐)\n"
                "96/128 = 更强表达能力, 但计算量增加",
    ),
    ParamDef(
        key="d_dims", label="解码器维度:", type=ParamType.INT, default=64,
        min_val=16, max_val=256, step=2, group=ParamGroup.ARCHITECTURE,
        tooltip="解码器基础通道维度, 控制生成细节丰富度。\n"
                "64 = 标准值(推荐)\n"
                "96/128 = 更精细细节, 但计算量增加",
    ),
    ParamDef(
        key="d_mask_dims", label="遮罩解码器维度:", type=ParamType.INT, default=22,
        min_val=8, max_val=128, step=2, group=ParamGroup.ARCHITECTURE,
        tooltip="遮罩解码器基础通道维度, 控制遮罩精细度。\n通常小于d_dims, 22是DFL默认值",
    ),
    ParamDef(
        key="freeze_encoder", label="冻结编码器", type=ParamType.BOOL, default=False,
        group=ParamGroup.ARCHITECTURE,
        tooltip="冻结编码器权重不更新。\n不建议开启, 除非你理解冻结的影响",
    ),
    ParamDef(
        key="freeze_inter", label="冻结中间层(DF)", type=ParamType.BOOL, default=False,
        group=ParamGroup.ARCHITECTURE,
        tooltip="冻结DF架构的Inter(瓶颈)层权重。\n仅对DF架构生效。\n不建议开启, 除非你理解冻结的影响",
    ),
    ParamDef(
        key="freeze_inter_AB", label="冻结Inter_AB(LIAE)", type=ParamType.BOOL, default=False,
        group=ParamGroup.ARCHITECTURE,
        tooltip="冻结LIAE架构的Inter_AB层权重。\n仅对LIAE架构生效。\n不建议开启, 除非你理解冻结的影响",
    ),
    ParamDef(
        key="freeze_inter_B", label="冻结Inter_B(LIAE)", type=ParamType.BOOL, default=False,
        group=ParamGroup.ARCHITECTURE,
        tooltip="冻结LIAE架构的Inter_B层权重。\n仅对LIAE架构生效。\n不建议开启, 除非你理解冻结的影响",
    ),
    ParamDef(
        key="freeze_decoder_mask", label="冻结遮罩解码器", type=ParamType.BOOL, default=False,
        group=ParamGroup.ARCHITECTURE,
        tooltip="冻结遮罩解码器权重不更新。\n不建议开启, 除非你理解冻结的影响",
    ),

    ParamDef(
        key="masked_training", label="遮罩训练", type=ParamType.BOOL, default=True,
        group=ParamGroup.FACE_DETAIL,
        tooltip="仅在人脸遮罩区域内计算损失。\n推荐开启, 避免背景区域干扰训练",
    ),
    ParamDef(
        key="eyes_mouth_prio", label="眼嘴优先", type=ParamType.BOOL, default=False,
        group=ParamGroup.FACE_DETAIL,
        tooltip="对眼睛和嘴巴区域施加更高权重(300倍)。\n"
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
        key="true_face_power", label="真实人脸强度:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=1.0, step=0.01, decimals=4, group=ParamGroup.LOSS_SAMPLING,
        tooltip="TrueFace损失强度, 让src编码分布向dst靠拢, 确保decoder_src能正确解码dst编码。\n"
                "0 = 不启用\n"
                "推荐: 0.01~0.1\n"
                "仅DF架构生效(LIAE无此功能)\n"
                "与FaceStyle目标相反, 同时开启可能冲突",
    ),
    ParamDef(
        key="gan_power", label="GAN强度:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=5.0, step=0.1, decimals=2, group=ParamGroup.LOSS_SAMPLING,
        tooltip="GAN对抗损失强度, 提升生成图像真实感和皮肤纹理。\n"
                "0 = 不启用(推荐初期)\n"
                "待重建稳定后(如50k iter)开启至0.1~1.0\n"
                "过高会导致伪影, 典型值0.1",
    ),
    ParamDef(
        key="gan_patch_size", label="GAN块大小:", type=ParamType.INT, default=16,
        min_val=3, max_val=640, step=1, group=ParamGroup.LOSS_SAMPLING,
        tooltip="GAN判别器的Patch大小。\n"
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
        key="vgg_perceptual_power", label="VGG感知损失:", type=ParamType.FLOAT, default=0.0,
        min_val=0.0, max_val=100.0, step=1.0, decimals=3, group=ParamGroup.LOSS_SAMPLING,
        tooltip="VGG感知损失强度, 基于VGG特征衡量图像感知相似度。\n"
                "0 = 不启用\n"
                "推荐: 1.0~10.0\n"
                "能显著改善视觉真实感, 减少'塑料感'\n"
                "比像素级损失更符合人眼感知",
    ),

    ParamDef(
        key="enable_torch_compile", label="torch.compile加速", type=ParamType.BOOL, default=True,
        group=ParamGroup.OPTIMIZATION,
        tooltip="启用torch.compile编译加速(PyTorch≥2.0)。\n"
                "首次迭代触发编译(约30~120秒), 后续迭代提速20%+\n"
                "编译失败自动降级为eager模式, 无副作用",
    ),
    ParamDef(
        key="use_ms_ssim", label="多尺度SSIM", type=ParamType.BOOL, default=True,
        group=ParamGroup.OPTIMIZATION,
        tooltip="使用多尺度SSIM(MS-SSIM)替代单尺度DSSIM。\n"
                "3尺度加权融合(0.5/0.3/0.2), 更好保留高频细节\n"
                "显存不足时自动降级为单尺度DSSIM",
    ),
    ParamDef(
        key="adaptive_mask_dilation", label="自适应Mask膨胀", type=ParamType.BOOL, default=True,
        group=ParamGroup.OPTIMIZATION,
        tooltip="自适应Mask膨胀(高斯模糊+形态学膨胀)。\n"
                "替代原始gaussian_blur+clamp, 边缘过渡更自然\n"
                "减少'切割感', 改善融合质量",
    ),
    ParamDef(
        key="mask_dilation_sigma", label="膨胀模糊sigma:", type=ParamType.FLOAT, default=2.0,
        min_val=0.5, max_val=10.0, step=0.5, decimals=1, group=ParamGroup.OPTIMIZATION,
        tooltip="Mask膨胀高斯模糊sigma值。\n"
                "越大模糊范围越宽, 边缘过渡越柔和\n"
                "仅adaptive_mask_dilation=True时生效",
    ),
    ParamDef(
        key="mask_dilation_radius", label="膨胀半径:", type=ParamType.INT, default=3,
        min_val=1, max_val=10, group=ParamGroup.OPTIMIZATION,
        tooltip="Mask形态学膨胀半径。\n"
                "越大膨胀范围越广\n"
                "仅adaptive_mask_dilation=True时生效",
    ),
    ParamDef(
        key="ramp_start_ratio", label="渐进GAN起始比:", type=ParamType.FLOAT, default=0.2,
        min_val=0.0, max_val=0.8, step=0.05, decimals=2, group=ParamGroup.OPTIMIZATION,
        tooltip="渐进式GAN起始比例。\n"
                "target_iter的此比例后线性开启GAN\n"
                "0.2 = 20%迭代后开始, 让模型先学好结构再细化纹理\n"
                "仅gan_power>0时生效, pretrain时强制为0",
    ),
    ParamDef(
        key="smart_stop_enabled", label="智能停止检测", type=ParamType.BOOL, default=True,
        group=ParamGroup.OPTIMIZATION,
        tooltip="智能停止检测, 训练收敛时建议停止或降低学习率。\n"
                "不会自动终止训练, 最终决策权在用户\n"
                "基于loss和预览SSIM双重收敛判定",
    ),
    ParamDef(
        key="smart_stop_window", label="收敛窗口:", type=ParamType.INT, default=500,
        min_val=100, max_val=5000, step=100, group=ParamGroup.OPTIMIZATION,
        tooltip="收敛检测滑动窗口步数。\n"
                "最近多少步内loss无显著改善则判定收敛\n"
                "仅smart_stop_enabled=True时生效",
    ),
    ParamDef(
        key="smart_stop_threshold", label="收敛阈值(%):", type=ParamType.FLOAT, default=0.1,
        min_val=0.001, max_val=5.0, step=0.01, decimals=3, group=ParamGroup.OPTIMIZATION,
        tooltip="收敛判定改善率阈值(%)。\n"
                "改善率低于此值判定为收敛\n"
                "仅smart_stop_enabled=True时生效",
    ),
]


def get_saehd_params_by_group(group: ParamGroup) -> list[ParamDef]:
    return [p for p in SAEHD_PARAM_DEFS if p.group == group]
