import torch
import torch.nn as nn
import torch.nn.functional as F

_FEAT_STRIDES = [8, 16, 32]
_NUM_ANCHORS = 2
_NUM_KPS = 5
_BASE_SIZES = [16, 64, 256]
_ANCHOR_SCALES = [1, 2]


def generate_anchors(input_size, strides=_FEAT_STRIDES, num_anchors=_NUM_ANCHORS,
                     base_sizes=_BASE_SIZES, scales=_ANCHOR_SCALES, device='cpu'):
    all_anchors = []
    num_per_level = []
    for stride, base_size in zip(strides, base_sizes):
        feat_h = input_size // stride
        feat_w = input_size // stride
        centers = torch.zeros(feat_h, feat_w, 2, device=device)
        for y in range(feat_h):
            for x in range(feat_w):
                centers[y, x, 0] = x * stride
                centers[y, x, 1] = y * stride
        centers = centers.reshape(-1, 2)

        anchors_per_level = []
        for scale in scales:
            size = base_size * scale
            half = size / 2.0
            a = torch.cat([
                centers[:, 0:1] - half, centers[:, 1:2] - half,
                centers[:, 0:1] + half, centers[:, 1:2] + half,
            ], dim=1)
            anchors_per_level.append(a)
        anchors = torch.cat(anchors_per_level, dim=0)
        all_anchors.append(anchors)
        num_per_level.append(anchors.shape[0])
    return all_anchors, num_per_level


def generate_anchors_fast(input_size, strides=_FEAT_STRIDES, num_anchors=_NUM_ANCHORS,
                          base_sizes=_BASE_SIZES, scales=_ANCHOR_SCALES, device='cpu'):
    all_anchors = []
    num_per_level = []
    for stride, base_size in zip(strides, base_sizes):
        feat_h = input_size // stride
        feat_w = input_size // stride
        ys = torch.arange(feat_h, device=device, dtype=torch.float32) * stride
        xs = torch.arange(feat_w, device=device, dtype=torch.float32) * stride
        cy, cx = torch.meshgrid(ys, xs, indexing='ij')
        centers = torch.stack([cx.flatten(), cy.flatten()], dim=1)

        anchors_per_level = []
        for scale in scales:
            size = base_size * scale
            half = size / 2.0
            a = torch.cat([
                centers[:, 0:1] - half, centers[:, 1:2] - half,
                centers[:, 0:1] + half, centers[:, 1:2] + half,
            ], dim=1)
            anchors_per_level.append(a)
        anchors = torch.cat(anchors_per_level, dim=0)
        all_anchors.append(anchors)
        num_per_level.append(anchors.shape[0])
    return all_anchors, num_per_level


def anchor_center(anchors):
    cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
    return torch.stack([cx, cy], dim=1)


def bbox_overlaps(b1, b2, is_aligned=True):
    if is_aligned:
        lt = torch.max(b1[:, :2], b2[:, :2])
        rb = torch.min(b1[:, 2:], b2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1 + area2 - overlap
        iou = overlap / union.clamp(min=1e-6)
        return iou
    else:
        lt = torch.max(b1[:, None, :2], b2[None, :, :2])
        rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1[:, None] + area2[None, :] - overlap
        iou = overlap / union.clamp(min=1e-6)
        return iou


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def kps2distance(points, kps):
    num_kps = kps.shape[-1] // 2
    dist = kps.clone()
    for i in range(num_kps):
        dist[..., 2 * i] = kps[..., 2 * i] - points[..., 0]
        dist[..., 2 * i + 1] = kps[..., 2 * i + 1] - points[..., 1]
    return dist


def quality_focal_loss(pred, label, score, beta=2.0):
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos], reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1)
    return loss


def diou_loss(pred, target, eps=1e-7):
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps
    ious = overlap / union

    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]
    c2 = cw ** 2 + ch ** 2 + eps

    left = ((target[:, 0] + target[:, 2]) - (pred[:, 0] + pred[:, 2])) ** 2 / 4
    right = ((target[:, 1] + target[:, 3]) - (pred[:, 1] + pred[:, 3])) ** 2 / 4
    rho2 = left + right

    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


class ATSSAssigner:
    def __init__(self, topk=9):
        self.topk = topk

    def assign(self, bboxes, num_level_bboxes, gt_bboxes, gt_labels=None):
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        overlaps = bbox_overlaps(bboxes, gt_bboxes, is_aligned=False)
        assigned_gt_inds = overlaps.new_full((num_bboxes,), 0, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            assigned_labels = None if gt_labels is None else overlaps.new_full(
                (num_bboxes,), -1, dtype=torch.long)
            return assigned_gt_inds, max_overlaps, assigned_labels

        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        gt_width = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_height = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        gt_area = torch.sqrt(torch.clamp(gt_width * gt_height, min=1e-4))

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt, device=overlaps.device)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        dist_min = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0]
        dist_min.div_(gt_area)
        is_in_gts = dist_min > 0.001
        is_pos = is_pos & is_in_gts

        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return assigned_gt_inds, max_overlaps, assigned_labels


class SCRFDLoss(nn.Module):
    def __init__(self, input_size=640, feat_strides=_FEAT_STRIDES,
                 num_anchors=_NUM_ANCHORS, num_classes=1, num_kps=_NUM_KPS,
                 topk=9, beta_qfl=2.0, bbox_loss_weight=2.0,
                 kps_loss_weight=0.1, kps_beta=1.0 / 9.0, use_qscore=False):
        super().__init__()
        self.input_size = input_size
        self.feat_strides = list(feat_strides)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_kps = num_kps
        self.beta_qfl = beta_qfl
        self.bbox_loss_weight = bbox_loss_weight
        self.kps_loss_weight = kps_loss_weight
        self.kps_beta = kps_beta
        self.use_qscore = use_qscore
        self.assigner = ATSSAssigner(topk=topk)
        self._anchor_list = None
        self._num_per_level = None

    def _get_anchors(self, device):
        if self._anchor_list is None or self._anchor_list[0].device != device:
            self._anchor_list, self._num_per_level = generate_anchors_fast(
                self.input_size, self.feat_strides, self.num_anchors,
                device=device)
        return self._anchor_list, self._num_per_level

    def forward(self, cls_scores, bbox_preds, kps_preds,
                gt_bboxes_list, gt_labels_list, gt_keypointss_list):
        device = cls_scores[0].device
        anchor_list, num_per_level = self._get_anchors(device)

        num_imgs = cls_scores[0].size(0)
        num_levels = len(cls_scores)

        total_loss_cls = 0.0
        total_loss_bbox = 0.0
        total_loss_kps = 0.0
        total_pos = 0.0

        for img_id in range(num_imgs):
            gt_bboxes = gt_bboxes_list[img_id]
            gt_labels = gt_labels_list[img_id]
            gt_keypointss = gt_keypointss_list[img_id]

            if gt_bboxes.numel() == 0:
                for lvl in range(num_levels):
                    cs = cls_scores[lvl][img_id].permute(1, 2, 0).reshape(-1, self.num_classes)
                    score = cs.new_zeros(cs.shape[0])
                    loss_cls = quality_focal_loss(cs, cs.new_full(cs.shape[0], self.num_classes, dtype=torch.long), score, beta=self.beta_qfl)
                    total_loss_cls = total_loss_cls + loss_cls.mean()
                continue

            flat_anchors = torch.cat(anchor_list)
            assigned_gt_inds, max_overlaps, assigned_labels = self.assigner.assign(
                flat_anchors, num_per_level, gt_bboxes, gt_labels)

            num_total_pos = (assigned_gt_inds > 0).sum().item()
            num_total_pos = max(num_total_pos, 1)

            bbox_targets = torch.zeros_like(flat_anchors)
            labels = flat_anchors.new_full((flat_anchors.size(0),), self.num_classes, dtype=torch.long)
            label_weights = flat_anchors.new_zeros(flat_anchors.size(0), dtype=torch.float)
            kps_targets = flat_anchors.new_zeros((flat_anchors.size(0), self.num_kps * 2))
            kps_weights = flat_anchors.new_zeros((flat_anchors.size(0), self.num_kps * 2))

            pos_inds = (assigned_gt_inds > 0).nonzero(as_tuple=False).squeeze(1)
            neg_inds = (assigned_gt_inds == 0).nonzero(as_tuple=False).squeeze(1)

            if len(pos_inds) > 0:
                pos_gt_inds = assigned_gt_inds[pos_inds] - 1
                bbox_targets[pos_inds] = gt_bboxes[pos_gt_inds]
                labels[pos_inds] = gt_labels[pos_gt_inds]
                label_weights[pos_inds] = 1.0
                kps_targets[pos_inds] = gt_keypointss[pos_gt_inds, :, :2].reshape(-1, self.num_kps * 2)
                kps_weights[pos_inds] = gt_keypointss[pos_gt_inds, :, 2].mean(dim=1, keepdim=True)
            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0

            for lvl in range(num_levels):
                if lvl == 0:
                    start = 0
                else:
                    start = sum(num_per_level[:lvl])
                end = start + num_per_level[lvl]
                stride = self.feat_strides[lvl]

                anchors = flat_anchors[start:end]
                lvl_labels = labels[start:end]
                lvl_label_weights = label_weights[start:end]
                lvl_bbox_targets = bbox_targets[start:end]
                lvl_kps_targets = kps_targets[start:end]
                lvl_kps_weights = kps_weights[start:end]

                cs = cls_scores[lvl][img_id].permute(1, 2, 0).reshape(-1, self.num_classes)
                bp = bbox_preds[lvl][img_id].permute(1, 2, 0).reshape(-1, 4)
                kp = kps_preds[lvl][img_id].permute(1, 2, 0).reshape(-1, self.num_kps * 2)

                pos = ((lvl_labels >= 0) & (lvl_labels < self.num_classes)).nonzero().squeeze(1)
                score = lvl_label_weights.new_zeros(lvl_labels.shape)

                if len(pos) > 0:
                    pos_bbox_targets = lvl_bbox_targets[pos]
                    pos_bbox_pred = bp[pos]
                    pos_anchors = anchors[pos]
                    pos_anchor_centers = anchor_center(pos_anchors) / stride

                    weight_targets = cs.detach().sigmoid()
                    weight_targets = weight_targets.max(dim=1)[0][pos]
                    pos_decode_bbox_targets = pos_bbox_targets / stride
                    pos_decode_bbox_pred = distance2bbox(pos_anchor_centers, pos_bbox_pred.clamp(min=0))

                    if self.use_qscore:
                        score[pos] = bbox_overlaps(
                            pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
                    else:
                        score[pos] = 1.0

                    loss_bbox = self.bbox_loss_weight * diou_loss(
                        pos_decode_bbox_pred, pos_decode_bbox_targets)
                    loss_bbox = (loss_bbox * weight_targets).sum()

                    pos_kps_targets = lvl_kps_targets[pos]
                    pos_kps_pred = kp[pos]
                    pos_kps_weights = lvl_kps_weights.max(dim=1)[0][pos] * weight_targets
                    pos_kps_weights = pos_kps_weights.reshape(-1, 1)

                    pos_decode_kps_targets = kps2distance(pos_anchor_centers, pos_kps_targets / stride)
                    pos_decode_kps_pred = pos_kps_pred

                    diff = (pos_decode_kps_pred * 1.0 - pos_decode_kps_targets * 1.0).abs()
                    beta = self.kps_beta
                    smooth_l1 = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
                    loss_kps = self.kps_loss_weight * smooth_l1.sum(dim=1)
                    loss_kps = (loss_kps * pos_kps_weights.squeeze(1)).sum()
                else:
                    loss_bbox = bp.sum() * 0
                    loss_kps = kp.sum() * 0

                loss_cls = quality_focal_loss(
                    cs, lvl_labels, score, beta=self.beta_qfl)
                loss_cls = (loss_cls * lvl_label_weights).sum() / num_total_pos

                total_loss_cls = total_loss_cls + loss_cls
                total_loss_bbox = total_loss_bbox + loss_bbox
                total_loss_kps = total_loss_kps + loss_kps
                total_pos += weight_targets.sum().item() if len(pos) > 0 else 0.0

        total_pos = max(total_pos, 1.0)
        total_loss_bbox = total_loss_bbox / total_pos
        total_loss_kps = total_loss_kps / total_pos

        return {
            'loss_cls': total_loss_cls,
            'loss_bbox': total_loss_bbox,
            'loss_kps': total_loss_kps,
            'loss': total_loss_cls + total_loss_bbox + total_loss_kps,
        }
