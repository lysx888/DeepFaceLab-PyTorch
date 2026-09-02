import math

import torch
import torch.nn as nn

_NUM_LANDMARKS = 106

_EYE_POINTS = [93, 96, 94, 95, 89, 90, 87, 91, 88, 92,
               35, 41, 40, 42, 39, 37, 33, 36, 34, 38]
_EYEBROW_POINTS = [101, 105, 104, 103, 102, 97, 98, 99, 100,
                   43, 48, 49, 51, 50, 46, 47, 45, 44]
_NOSE_POINTS = [72, 73, 74, 86, 75, 76, 77, 78, 79, 80, 85, 84, 83, 82, 81]
_MOUTH_POINTS = [65, 66, 62, 70, 69, 57, 60, 54,
                 52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]
_CONTOUR_POINTS = [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0,
                   24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17]


def _build_region_weights() -> torch.Tensor:
    weights = torch.ones(_NUM_LANDMARKS, dtype=torch.float32)
    for idx in _EYE_POINTS:
        weights[idx] = 2.0
    for idx in _MOUTH_POINTS:
        weights[idx] = 2.0
    for idx in _EYEBROW_POINTS:
        weights[idx] = 1.5
    for idx in _NOSE_POINTS:
        weights[idx] = 1.5
    for idx in _CONTOUR_POINTS:
        weights[idx] = 1.2
    return weights


_REGION_WEIGHTS = _build_region_weights()


class WingLoss(nn.Module):
    def __init__(self, w: float = 0.5, epsilon: float = 0.1):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1.0 + w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - self.C
        )
        return loss


class AdaptiveWingLoss(nn.Module):
    def __init__(self, w: float = 0.5, theta: float = 0.05, alpha: float = 0.1):
        super().__init__()
        self.w = w
        self.theta = theta
        self.alpha = alpha
        self.C = theta - w * math.log(1.0 + theta / alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.theta,
            self.w * torch.log(1.0 + diff / self.alpha),
            diff - self.C
        )
        return loss


class IFLandmarkLoss(nn.Module):
    def __init__(self, loss_type: str = "wing", warmup_ratio: float = 0.2,
                 blend_ratio: float = 0.05):
        super().__init__()
        self.loss_type = loss_type
        self.warmup_ratio = warmup_ratio
        self.blend_ratio = blend_ratio
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        if loss_type == "awing":
            self.main_loss = AdaptiveWingLoss(w=0.5, theta=0.05, alpha=0.1)
        else:
            self.main_loss = WingLoss(w=0.5, epsilon=0.1)
        self.register_buffer('region_weights', _REGION_WEIGHTS.repeat_interleave(2).view(1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                progress: float = 1.0,
                visible: torch.Tensor | None = None) -> torch.Tensor:
        warmup_end = self.warmup_ratio
        blend_start = warmup_end - self.blend_ratio

        if progress < blend_start:
            loss = self.smooth_l1(pred, target)
        elif progress < warmup_end:
            alpha = (progress - blend_start) / self.blend_ratio
            sl1 = self.smooth_l1(pred, target)
            wing = self.main_loss(pred, target)
            loss = (1.0 - alpha) * sl1 + alpha * wing
        else:
            loss = self.main_loss(pred, target)

        if visible is None:
            mask = self.region_weights.expand(loss.shape[0], -1)
        else:
            vis_expanded = visible.repeat_interleave(2, dim=1)
            mask = self.region_weights * vis_expanded

        loss = loss * mask
        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom
