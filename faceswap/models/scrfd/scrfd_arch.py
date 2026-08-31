import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from faceswap.shared.logger import get_logger

_logger = get_logger("scrfd_arch")

_FEAT_STRIDES = [8, 16, 32]
_NUM_ANCHORS = 2
_NUM_KPS = 5


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _make_downsample(inplanes, planes, stride, avg_down):
    layers = []
    conv_stride = stride
    if avg_down:
        conv_stride = 1
        layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
    layers.append(nn.Conv2d(inplanes, planes, 1, stride=conv_stride, bias=False))
    layers.append(nn.BatchNorm2d(planes))
    return nn.Sequential(*layers)


def _make_res_layer(inplanes, planes, num_blocks, stride=1, avg_down=False):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = _make_downsample(inplanes, planes, stride, avg_down)
    layers = [BasicBlock(inplanes, planes, stride=stride, downsample=downsample)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(planes, planes))
    return nn.Sequential(*layers)


class ResNetV1e(nn.Module):
    def __init__(self, base_channels=56, stage_blocks=(3, 4, 2, 3), stage_planes=(56, 88, 88, 224)):
        super().__init__()
        sc = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, sc // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(sc // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(sc // 2, sc // 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(sc // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(sc // 2, sc, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        inplanes = sc
        self.res_layers = nn.ModuleList()
        for i, num_blocks in enumerate(stage_blocks):
            planes = stage_planes[i]
            s = 1 if i == 0 else 2
            layer = _make_res_layer(inplanes, planes, num_blocks, stride=s, avg_down=True)
            self.res_layers.append(layer)
            inplanes = planes

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        outs = []
        for layer in self.res_layers:
            x = layer(x)
            outs.append(x)
        return tuple(outs)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PAFPN(nn.Module):
    def __init__(self, in_channels=(56, 88, 88, 224), out_channels=56, start_level=1, num_outs=3):
        super().__init__()
        self.start_level = start_level
        self.num_outs = num_outs
        used = len(in_channels) - start_level

        self.lateral_convs = nn.ModuleList()
        for i in range(used):
            self.lateral_convs.append(nn.Conv2d(in_channels[start_level + i], out_channels, 1, bias=True))

        self.fpn_convs = nn.ModuleList()
        for i in range(num_outs):
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True))

        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(used - 1):
            self.downsample_convs.append(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=True))
            self.pafpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True))

    def forward(self, inputs):
        laterals = [self.lateral_convs[i](inputs[i + self.start_level]) for i in range(len(self.lateral_convs))]

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='nearest')

        inter_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        for i in range(len(inter_outs) - 1):
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[i](inter_outs[i])

        outs = [inter_outs[0]]
        for i in range(1, len(inter_outs)):
            outs.append(self.pafpn_convs[i - 1](inter_outs[i]))

        return tuple(outs)


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return x * self.scale


class SCRFDHead(nn.Module):
    def __init__(self, in_channels=56, feat_channels=80, stacked_convs=3,
                 num_anchors=2, num_classes=1, num_kps=5,
                 feat_strides=(8, 16, 32)):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_kps = num_kps
        self.feat_strides = list(feat_strides)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kps_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.kps_preds = nn.ModuleList()
        self.scales = nn.ModuleList()

        for _ in self.feat_strides:
            cls_convs = nn.ModuleList()
            for i in range(stacked_convs):
                chn = in_channels if i == 0 else feat_channels
                cls_convs.append(ConvBNReLU(chn, feat_channels))
            self.cls_convs.append(cls_convs)

            self.reg_convs.append(cls_convs)
            self.kps_convs.append(cls_convs)

            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes * num_anchors, 3, padding=1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4 * num_anchors, 3, padding=1))
            self.kps_preds.append(nn.Conv2d(feat_channels, num_kps * 2 * num_anchors, 3, padding=1))
            self.scales.append(Scale(1.0))

        _bias_cls = -math.log((1 - 0.01) / 0.01)
        for cls_convs in self.cls_convs:
            for m in cls_convs:
                nn.init.normal_(m.conv.weight, std=0.01)
        for conv in self.cls_preds:
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, _bias_cls)
        for conv in self.reg_preds:
            nn.init.normal_(conv.weight, std=0.01)
        for conv in self.kps_preds:
            nn.init.normal_(conv.weight, std=0.01)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        kps_preds = []
        for i, feat in enumerate(feats):
            cls_feat = feat
            for conv in self.cls_convs[i]:
                cls_feat = conv(cls_feat)
            reg_feat = cls_feat

            cls_score = self.cls_preds[i](cls_feat)
            bbox_pred = self.scales[i](self.reg_preds[i](reg_feat))
            kps_pred = self.kps_preds[i](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            kps_preds.append(kps_pred)
        return cls_scores, bbox_preds, kps_preds


class SCRFDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetV1e(
            base_channels=56,
            stage_blocks=(3, 4, 2, 3),
            stage_planes=(56, 88, 88, 224),
        )
        self.neck = PAFPN(
            in_channels=(56, 88, 88, 224),
            out_channels=56,
            start_level=1,
            num_outs=3,
        )
        self.head = SCRFDHead(
            in_channels=56,
            feat_channels=80,
            stacked_convs=3,
            num_anchors=_NUM_ANCHORS,
            num_classes=1,
            num_kps=_NUM_KPS,
            feat_strides=_FEAT_STRIDES,
        )

    def forward(self, x):
        backbone_outs = self.backbone(x)
        neck_outs = self.neck(backbone_outs)
        cls_scores, bbox_preds, kps_preds = self.head(neck_outs)
        return cls_scores, bbox_preds, kps_preds

    def load_pretrained_onnx(self, onnx_path: str) -> int:
        import onnx
        m = onnx.load(onnx_path)
        onnx_inits = {init.name: onnx.numpy_helper.to_array(init) for init in m.graph.initializer}
        state_dict = self.state_dict()
        loaded = 0

        def _load_bn_folded(onnx_w_id: str, onnx_b_id: str, pt_conv: str, pt_bn: str):
            nonlocal loaded
            w = onnx_inits.get(onnx_w_id)
            b = onnx_inits.get(onnx_b_id)
            if w is not None and pt_conv + '.weight' in state_dict:
                state_dict[pt_conv + '.weight'] = torch.from_numpy(w.copy())
                loaded += 1
            if b is not None and pt_bn + '.bias' in state_dict:
                state_dict[pt_bn + '.weight'] = torch.ones_like(state_dict[pt_bn + '.weight'])
                state_dict[pt_bn + '.bias'] = torch.from_numpy(b.copy())
                state_dict[pt_bn + '.running_mean'] = torch.zeros_like(state_dict[pt_bn + '.running_mean'])
                state_dict[pt_bn + '.running_var'] = torch.ones_like(state_dict[pt_bn + '.running_var'])
                loaded += 1

        def _load_plain_conv(onnx_w_name: str, onnx_b_name: str, pt_conv: str):
            nonlocal loaded
            w = onnx_inits.get(onnx_w_name)
            b = onnx_inits.get(onnx_b_name) if onnx_b_name else None
            if w is not None and pt_conv + '.weight' in state_dict:
                state_dict[pt_conv + '.weight'] = torch.from_numpy(w.copy())
                loaded += 1
            if b is not None and pt_conv + '.bias' in state_dict:
                state_dict[pt_conv + '.bias'] = torch.from_numpy(b.copy())
                loaded += 1
            elif pt_conv + '.bias' in state_dict:
                state_dict[pt_conv + '.bias'] = torch.zeros_like(state_dict[pt_conv + '.bias'])

        def _load_scalar(onnx_name: str, pt_name: str):
            nonlocal loaded
            v = onnx_inits.get(onnx_name)
            if v is not None and pt_name in state_dict:
                state_dict[pt_name] = torch.from_numpy(v.copy())
                loaded += 1

        _BACKBONE_BN_FOLDED = [
            ('547', '549', 'backbone.stem.0', 'backbone.stem.1'),
            ('551', '553', 'backbone.stem.3', 'backbone.stem.4'),
            ('555', '557', 'backbone.stem.6', 'backbone.stem.7'),
            ('559', '561', 'backbone.res_layers.0.0.conv1', 'backbone.res_layers.0.0.bn1'),
            ('563', '565', 'backbone.res_layers.0.0.conv2', 'backbone.res_layers.0.0.bn2'),
            ('567', '569', 'backbone.res_layers.0.1.conv1', 'backbone.res_layers.0.1.bn1'),
            ('571', '573', 'backbone.res_layers.0.1.conv2', 'backbone.res_layers.0.1.bn2'),
            ('575', '577', 'backbone.res_layers.0.2.conv1', 'backbone.res_layers.0.2.bn1'),
            ('579', '581', 'backbone.res_layers.0.2.conv2', 'backbone.res_layers.0.2.bn2'),
            ('583', '585', 'backbone.res_layers.1.0.conv1', 'backbone.res_layers.1.0.bn1'),
            ('587', '589', 'backbone.res_layers.1.0.conv2', 'backbone.res_layers.1.0.bn2'),
            ('591', '593', 'backbone.res_layers.1.0.downsample.1', 'backbone.res_layers.1.0.downsample.2'),
            ('595', '597', 'backbone.res_layers.1.1.conv1', 'backbone.res_layers.1.1.bn1'),
            ('599', '601', 'backbone.res_layers.1.1.conv2', 'backbone.res_layers.1.1.bn2'),
            ('603', '605', 'backbone.res_layers.1.2.conv1', 'backbone.res_layers.1.2.bn1'),
            ('607', '609', 'backbone.res_layers.1.2.conv2', 'backbone.res_layers.1.2.bn2'),
            ('611', '613', 'backbone.res_layers.1.3.conv1', 'backbone.res_layers.1.3.bn1'),
            ('615', '617', 'backbone.res_layers.1.3.conv2', 'backbone.res_layers.1.3.bn2'),
            ('619', '621', 'backbone.res_layers.2.0.conv1', 'backbone.res_layers.2.0.bn1'),
            ('623', '625', 'backbone.res_layers.2.0.conv2', 'backbone.res_layers.2.0.bn2'),
            ('627', '629', 'backbone.res_layers.2.0.downsample.1', 'backbone.res_layers.2.0.downsample.2'),
            ('631', '633', 'backbone.res_layers.2.1.conv1', 'backbone.res_layers.2.1.bn1'),
            ('635', '637', 'backbone.res_layers.2.1.conv2', 'backbone.res_layers.2.1.bn2'),
            ('639', '641', 'backbone.res_layers.3.0.conv1', 'backbone.res_layers.3.0.bn1'),
            ('643', '645', 'backbone.res_layers.3.0.conv2', 'backbone.res_layers.3.0.bn2'),
            ('647', '649', 'backbone.res_layers.3.0.downsample.1', 'backbone.res_layers.3.0.downsample.2'),
            ('651', '653', 'backbone.res_layers.3.1.conv1', 'backbone.res_layers.3.1.bn1'),
            ('655', '657', 'backbone.res_layers.3.1.conv2', 'backbone.res_layers.3.1.bn2'),
            ('659', '661', 'backbone.res_layers.3.2.conv1', 'backbone.res_layers.3.2.bn1'),
            ('663', '665', 'backbone.res_layers.3.2.conv2', 'backbone.res_layers.3.2.bn2'),
        ]
        for ow, ob, pc, pb in _BACKBONE_BN_FOLDED:
            _load_bn_folded(ow, ob, pc, pb)

        _HEAD_BN_FOLDED = [
            ('667', '669', 'head.cls_convs.0.0.conv', 'head.cls_convs.0.0.bn'),
            ('671', '673', 'head.cls_convs.0.1.conv', 'head.cls_convs.0.1.bn'),
            ('675', '677', 'head.cls_convs.0.2.conv', 'head.cls_convs.0.2.bn'),
            ('679', '681', 'head.cls_convs.1.0.conv', 'head.cls_convs.1.0.bn'),
            ('683', '685', 'head.cls_convs.1.1.conv', 'head.cls_convs.1.1.bn'),
            ('687', '689', 'head.cls_convs.1.2.conv', 'head.cls_convs.1.2.bn'),
            ('691', '693', 'head.cls_convs.2.0.conv', 'head.cls_convs.2.0.bn'),
            ('695', '697', 'head.cls_convs.2.1.conv', 'head.cls_convs.2.1.bn'),
            ('699', '701', 'head.cls_convs.2.2.conv', 'head.cls_convs.2.2.bn'),
        ]
        for ow, ob, pc, pb in _HEAD_BN_FOLDED:
            _load_bn_folded(ow, ob, pc, pb)

        _NECK_PLAIN = [
            ('neck.lateral_convs.0.conv.weight', 'neck.lateral_convs.0.conv.bias', 'neck.lateral_convs.0'),
            ('neck.lateral_convs.1.conv.weight', 'neck.lateral_convs.1.conv.bias', 'neck.lateral_convs.1'),
            ('neck.lateral_convs.2.conv.weight', 'neck.lateral_convs.2.conv.bias', 'neck.lateral_convs.2'),
            ('neck.fpn_convs.0.conv.weight', 'neck.fpn_convs.0.conv.bias', 'neck.fpn_convs.0'),
            ('neck.fpn_convs.1.conv.weight', None, 'neck.fpn_convs.1'),
            ('neck.fpn_convs.2.conv.weight', None, 'neck.fpn_convs.2'),
            ('neck.downsample_convs.0.conv.weight', 'neck.downsample_convs.0.conv.bias', 'neck.downsample_convs.0'),
            ('neck.downsample_convs.1.conv.weight', 'neck.downsample_convs.1.conv.bias', 'neck.downsample_convs.1'),
            ('neck.pafpn_convs.0.conv.weight', 'neck.pafpn_convs.0.conv.bias', 'neck.pafpn_convs.0'),
            ('neck.pafpn_convs.1.conv.weight', 'neck.pafpn_convs.1.conv.bias', 'neck.pafpn_convs.1'),
        ]
        for ow, ob, pc in _NECK_PLAIN:
            _load_plain_conv(ow, ob, pc)

        ds0_bias = onnx_inits.get('neck.downsample_convs.0.conv.bias')
        ds1_bias = onnx_inits.get('neck.downsample_convs.1.conv.bias')
        if ds0_bias is not None and 'neck.fpn_convs.1.bias' in state_dict:
            state_dict['neck.fpn_convs.1.bias'] = torch.from_numpy(ds0_bias.copy())
            loaded += 1
        if ds1_bias is not None and 'neck.fpn_convs.2.bias' in state_dict:
            state_dict['neck.fpn_convs.2.bias'] = torch.from_numpy(ds1_bias.copy())
            loaded += 1

        _HEAD_PREDS = [
            ('bbox_head.stride_cls.(8, 8).weight', 'bbox_head.stride_cls.(8, 8).bias', 'head.cls_preds.0'),
            ('bbox_head.stride_cls.(16, 16).weight', 'bbox_head.stride_cls.(16, 16).bias', 'head.cls_preds.1'),
            ('bbox_head.stride_cls.(32, 32).weight', 'bbox_head.stride_cls.(32, 32).bias', 'head.cls_preds.2'),
            ('bbox_head.stride_reg.(8, 8).weight', 'bbox_head.stride_reg.(8, 8).bias', 'head.reg_preds.0'),
            ('bbox_head.stride_reg.(16, 16).weight', 'bbox_head.stride_reg.(16, 16).bias', 'head.reg_preds.1'),
            ('bbox_head.stride_reg.(32, 32).weight', 'bbox_head.stride_reg.(32, 32).bias', 'head.reg_preds.2'),
            ('bbox_head.stride_kps.(8, 8).weight', 'bbox_head.stride_kps.(8, 8).bias', 'head.kps_preds.0'),
            ('bbox_head.stride_kps.(16, 16).weight', 'bbox_head.stride_kps.(16, 16).bias', 'head.kps_preds.1'),
            ('bbox_head.stride_kps.(32, 32).weight', 'bbox_head.stride_kps.(32, 32).bias', 'head.kps_preds.2'),
        ]
        for ow, ob, pc in _HEAD_PREDS:
            _load_plain_conv(ow, ob, pc)

        for i in range(3):
            _load_scalar(f'bbox_head.scales.{i}.scale', f'head.scales.{i}.scale')

        self.load_state_dict(state_dict)
        _logger.info(f"从ONNX加载 {loaded} 个预训练权重: {onnx_path}")
        return loaded

    def export_onnx(self, path: str, input_size: int = 640) -> None:
        was_training = self.training
        self.eval()

        class _ExportWrapper(nn.Module):
            def __init__(self, scrfd):
                super().__init__()
                self.scrfd = scrfd

            def forward(self, x):
                cls_scores, bbox_preds, kps_preds = self.scrfd(x)
                scores_outs = []
                bbox_outs = []
                kps_outs = []
                for i in range(len(cls_scores)):
                    cs = cls_scores[i].permute(0, 2, 3, 1).reshape(1, -1, 1).sigmoid()
                    bp = bbox_preds[i].permute(0, 2, 3, 1).reshape(1, -1, 4)
                    kp = kps_preds[i].permute(0, 2, 3, 1).reshape(1, -1, 10)
                    scores_outs.append(cs)
                    bbox_outs.append(bp)
                    kps_outs.append(kp)
                return tuple(scores_outs + bbox_outs + kps_outs)

        wrapper = _ExportWrapper(self)
        wrapper.eval()
        device = next(self.parameters()).device
        dummy = torch.randn(1, 3, input_size, input_size, device=device)
        torch.onnx.export(
            wrapper, dummy, path,
            input_names=["input.1"],
            output_names=[f"output_{i}" for i in range(9)],
            opset_version=18,
            external_data=False,
            dynamic_axes={
                "input.1": {2: "height", 3: "width"},
            },
            dynamo=False,
        )
        if was_training:
            self.train()
        _logger.info(f"SCRFD ONNX exported: {path} (dynamic input)")


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets, iou_thresh=0.4):
    if len(dets) == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def scrfd_detect(net, img, input_size=640, det_thresh=0.5, nms_thresh=0.4):
    net.eval()
    device = next(net.parameters()).device

    im_ratio = float(img.shape[0]) / img.shape[1]
    if im_ratio > 1.0:
        new_height = input_size
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img

    blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128.0, (input_size, input_size), (127.5, 127.5, 127.5), swapRB=True)
    blob_tensor = torch.from_numpy(blob).to(device)

    with torch.no_grad():
        cls_scores, bbox_preds, kps_preds = net(blob_tensor)

    scores_list = []
    bboxes_list = []
    kpss_list = []

    for idx, stride in enumerate(_FEAT_STRIDES):
        scores = cls_scores[idx].squeeze(0).permute(1, 2, 0).reshape(-1, 1).sigmoid().cpu().numpy()
        bbox_pred = bbox_preds[idx].squeeze(0).permute(1, 2, 0).reshape(-1, 4).cpu().numpy() * stride
        kps_pred = kps_preds[idx].squeeze(0).permute(1, 2, 0).reshape(-1, 10).cpu().numpy() * stride

        height = input_size // stride
        width = input_size // stride
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if _NUM_ANCHORS > 1:
            anchor_centers = np.stack([anchor_centers] * _NUM_ANCHORS, axis=1).reshape((-1, 2))

        pos_inds = np.where(scores[:, 0] >= det_thresh)[0]
        bboxes = distance2bbox(anchor_centers, bbox_pred)
        pos_scores = scores[pos_inds, 0]
        pos_bboxes = bboxes[pos_inds]
        kpss = distance2kps(anchor_centers, kps_pred)
        kpss = kpss.reshape((kpss.shape[0], -1, 2))

        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)
        kpss_list.append(kpss[pos_inds])

    if not scores_list or all(len(s) == 0 for s in scores_list):
        return np.empty((0, 5)), np.empty((0, 5, 2))

    scores = np.concatenate(scores_list)
    bboxes = np.concatenate(bboxes_list) / det_scale
    kpss = np.concatenate(kpss_list) / det_scale

    dets = np.hstack((bboxes, scores[:, None]))
    keep = nms(dets, nms_thresh)
    dets = dets[keep, :]
    kpss = kpss[keep, :]
    return dets, kpss
