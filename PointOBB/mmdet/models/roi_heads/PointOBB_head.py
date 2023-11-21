import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, rbbox2roi, build_assigner, build_sampler, multi_apply
from ..builder import HEADS, MODELS, build_head, build_roi_extractor, build_loss
from .standard_roi_head import StandardRoIHead
from .cascade_roi_head import CascadeRoIHead
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core.bbox.transforms import rbbox2result

import copy
import numpy as np
import cv2
from mmcv.cnn import Scale, ConvModule
from mmcv.ops import box_iou_rotated
from typing import Any, List, Sequence, Tuple, Union
from torch import Tensor
from mmdet.models.utils.base_bbox_coder import BaseBBoxCoder
from ..detectors.utils import obb2xyxy, regularize_boxes, reduce_mean, obb2poly_np


RangeType = Sequence[Tuple[int, int]]
INF = 1e8

def meshgrid(x: Tensor,
             y: Tensor,
             row_major: bool = True) -> Tuple[Tensor, Tensor]:
    yy, xx = torch.meshgrid(y, x)
    if row_major:
        # warning .flatten() would cause error in ONNX exportingF
        # have to use reshape here
        return xx.reshape(-1), yy.reshape(-1)
    else:
        return yy.reshape(-1), xx.reshape(-1)
        
def obb2cxcywh_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    wh = bias * 2
    return torch.cat([center, wh, torch.zeros_like(theta)], dim=-1)

@HEADS.register_module()
class PSCCoder(BaseBBoxCoder):
    """Phase-Shifting Coder.

    `Phase-Shifting Coder (PSC)
    <https://arxiv.org/abs/2211.06368>`.

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        dual_freq (bool, optional): Use dual frequency. Default: True.
        num_step (int, optional): Number of phase steps. Default: 3.
        thr_mod (float): Threshold of modulation. Default: 0.47.
    """

    def __init__(self,
                 angle_version: str,
                 dual_freq: bool = True,
                 num_step: int = 3,
                 thr_mod: float = 0.47):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.dual_freq = dual_freq
        self.num_step = num_step
        self.thr_mod = thr_mod
        if self.dual_freq:
            self.encode_size = 2 * self.num_step
        else:
            self.encode_size = self.num_step

        self.coef_sin = torch.tensor(
            tuple(
                torch.sin(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))
        self.coef_cos = torch.tensor(
            tuple(
                torch.cos(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Phase-Shifting Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
        """
        phase_targets = angle_targets * 2
        phase_shift_targets = tuple(
            torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
            for x in range(self.num_step))

        # Dual-freq PSC for square-like problem
        if self.dual_freq:
            phase_targets = angle_targets * 4
            phase_shift_targets += tuple(
                torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
                for x in range(self.num_step))

        return torch.cat(phase_shift_targets, axis=-1)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Phase-Shifting Decoder.

        Args:
            angle_preds (Tensor): The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """
        self.coef_sin = self.coef_sin.to(angle_preds)
        self.coef_cos = self.coef_cos.to(angle_preds)

        phase_sin = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_sin,
            dim=-1,
            keepdim=keepdim)
        phase_cos = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_cos,
            dim=-1,
            keepdim=keepdim)
        phase_mod = phase_cos**2 + phase_sin**2
        phase = -torch.atan2(phase_sin, phase_cos)  # In range [-pi,pi)

        if self.dual_freq:
            phase_sin = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_sin,
                dim=-1,
                keepdim=keepdim)
            phase_cos = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_cos,
                dim=-1,
                keepdim=keepdim)
            phase_mod = phase_cos**2 + phase_sin**2
            phase2 = -torch.atan2(phase_sin, phase_cos) / 2

            # Phase unwarpping, dual freq mixing
            # Angle between phase and phase2 is obtuse angle
            idx = torch.cos(phase) * torch.cos(phase2) + torch.sin(
                phase) * torch.sin(phase2) < 0
            # Add pi to phase2 and keep it in range [-pi,pi)
            phase2[idx] = phase2[idx] % (2 * math.pi) - math.pi
            phase = phase2

        # Set the angle of isotropic objects to zero
        phase[phase_mod < self.thr_mod] *= 0
        angle_pred = phase / 2
        return angle_pred

@HEADS.register_module()
class PointOBBHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, 
                 bbox_roi_extractor, 
                 num_stages, 
                 bbox_head, 
                 top_k=7, 
                 with_atten=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 scale_angle: bool = True,
                 stacked_convs = 4,
                 loss_symmetry_ss=dict(
                     type='SmoothL1Loss', loss_weight=1.0, beta=0.1),
                 angle_coder=dict(
                    type='PSCCoder',
                    angle_version='le90',
                    dual_freq=False,
                    num_step=3,
                    thr_mod=0),
                 angle_version = 'le90',
                 use_angle_loss = True,
                 add_angle_pred_begin = False,
                 not_use_rot_mil = False,
                 detach_angle_head = False,
                 rotation_agnostic_classes = None,
                 agnostic_resize_classes = None,
                 cls_scores_weight = 1.0,
                 ins_scores_weight = 1.0,
                 **kwargs):
        super(PointOBBHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head, **kwargs)
        self.threshold = 0.3
        self.merge_mode = 'weighted_clsins'
        self.test_mean_iou = False
        # self.test_mean_iou = True

        self.sum_iou = 0
        self.sum_num = 0
        self.num_stages = num_stages
        self.topk1 = top_k  # 7
        self.topk2 = top_k  # 7

        self.featmap_strides = bbox_roi_extractor.featmap_strides
        self.with_atten = with_atten
        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.in_channels=256 
        self.feat_channels=256
        self.stacked_convs=stacked_convs
        
        self.is_scale_angle = scale_angle
        self.angle_coder = HEADS.build(angle_coder)
        self.loss_symmetry_ss = build_loss(loss_symmetry_ss)
        self.angle_version = angle_version
        self.rotation_agnostic_classes = rotation_agnostic_classes
        self.agnostic_resize_classes = agnostic_resize_classes
        self.add_angle_pred_begin = add_angle_pred_begin
        self.use_angle_loss = use_angle_loss
        self.not_use_rot_mil = not_use_rot_mil
        self.detach_angle_head = detach_angle_head
        self.cls_scores_weight = cls_scores_weight
        self.ins_scores_weight = ins_scores_weight
        self.num_classes = self.bbox_head.num_classes
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)
    
    def angle_forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        angle_results = []
        for feat in feats:
            if self.detach_angle_head:
                feat_detach = feat.clone().detach()
                single_angle_pred = self.angle_forward_single(feat_detach)
            else:
                single_angle_pred = self.angle_forward_single(feat)
            angle_results.append(single_angle_pred)

        return tuple(angle_results)
    
    def angle_forward_single(self, x: Tensor):
        cls_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        # cls_score = self.conv_cls(cls_feat)
        angle_pred = self.conv_angle(cls_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        return angle_pred

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        # self.cdb = build_head(dict(type='ConvConcreteDB', cfg=None, planes=256))
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def grid_priors(self,
                    featmap_sizes: List[Tuple],
                    dtype: torch.dtype = torch.float32,
                    device = 'cuda',
                    with_stride: bool = False):
        num_levels = len(self.featmap_strides)
        assert num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors
    
    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int],
                                 level_idx: int,
                                 dtype: torch.dtype = torch.float32,
                                 device = 'cuda',
                                 offset = 0.5,
                                 with_stride: bool = False) -> Tensor:
        feat_h, feat_w = featmap_size
        stride_w = self.featmap_strides[level_idx]
        stride_h = stride_w
        shift_x = ((torch.arange(0, feat_w, device=device) + offset) * stride_w).to(dtype)
        shift_y = ((torch.arange(0, feat_h, device=device) + offset) * stride_h).to(dtype)
        shift_xx, shift_yy = meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                         stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                         stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points

    def get_targets(self, x, points, gt_points, proposals, gt_labels, img_metas):
        self.norm_on_bbox = True
        num_levels = len(x)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        labels_list, angle_targets_list, id_targets_list = multi_apply(
                self._get_targets_single,
                gt_points,
                proposals,
                gt_labels,
                img_metas,
                points=concat_points,
                num_points_per_lvl=num_points)
        
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]
        id_targets_list = [
            id_targets.split(num_points, 0) for id_targets in id_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_angle_targets = []
        concat_lvl_id_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            # bbox_targets = torch.cat(
            #     [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            id_targets = torch.cat(
                [id_targets[i] for id_targets in id_targets_list])
        
            concat_lvl_angle_targets.append(angle_targets)
            concat_lvl_id_targets.append(id_targets)
 
        return (concat_lvl_labels, concat_lvl_angle_targets, concat_lvl_id_targets)
    
    def _get_targets_single(
            self, gt_points, proposals, gt_label, img_meta, points,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        self.center_sampling = True
        self.center_sample_radius = 1.5
        self.pseudow = 3
        self.pseudoh = 2

        num_points = points.size(0)
        num_gts = len(gt_points)
        gt_labels = gt_label
        gt_bid = img_meta['gt_bid']
        gen_proposals = proposals.reshape(len(gt_points), -1, proposals.size(-1))
        if gt_points.size(-1) == 2:
            extra_tensor = torch.tensor([self.pseudow, self.pseudoh, gen_proposals[0,0,-1]], 
                                        device=gt_points.device, dtype=gt_points.dtype).repeat(len(gt_points), 1)
            gt_bboxes = torch.cat((gt_points, extra_tensor), dim=1)
        else:
            gt_bboxes = gt_points.clone()

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 1)), \
                   gt_bboxes.new_zeros((num_points,))

        areas = (gt_bboxes[:,2] * gt_bboxes[:,3]).squeeze() 
        gt_bboxes = regularize_boxes(gt_bboxes, pattern=self.angle_version)

        areas = areas[None].repeat(num_points, 1)
        points = points[:, None, :].expand(num_points, num_gts, 2) 
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)
        if self.center_sampling:
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.featmap_strides[lvl_idx] * radius
                lvl_begin = lvl_end
            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = inside_center_bbox_mask

        areas[inside_gt_bbox_mask == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        angle_targets = gt_angle[range(num_points), min_area_inds]
        bid_targets = gt_bid[min_area_inds]
        return labels, angle_targets, bid_targets
    
    def _get_rotation_agnostic_mask(self, cls):
        _rot_agnostic_mask = torch.zeros_like(cls, dtype=torch.bool)
        for c in self.rotation_agnostic_classes:
            _rot_agnostic_mask = torch.logical_or(_rot_agnostic_mask, cls == c)
        return _rot_agnostic_mask

    def GetCompacted(self, dtype, pos_angle_preds, pos_angle_targets, pos_bid_targets, pos_labels):
        bid, idx = torch.unique(pos_bid_targets, return_inverse=True)  
        device = bid.device
        compacted_bid_targets = torch.empty_like(bid)
        compacted_angle_targets = torch.empty_like(bid)
        for i in range(len(bid)):
            mask = (idx == i)
            if mask.sum() > 0:
                compacted_bid_targets[i] = pos_bid_targets[mask].mean(dim=0)
                compacted_angle_targets[i] = pos_angle_targets[:, 0][mask].mean(dim=0)
        b_flp = (compacted_bid_targets % 1 > 0.5).sum() > 0
        _, bidx, bcnt = torch.unique(compacted_bid_targets.long(), return_inverse=True, return_counts=True)
        bmsk = bcnt[bidx] == 2
        compacted_angle_targets = compacted_angle_targets[bmsk].view(-1, 2)
        # angle preds
        compacted_angle_preds = torch.empty((*bid.shape, pos_angle_preds.shape[-1]),
                                                device=device, dtype=dtype)
        for i in range(compacted_angle_preds.size(0)):
            mask = (idx == i)
            if mask.sum() == 0:
                compacted_angle_preds[i] = 0
            else:  # 'mean'
                compacted_angle_preds[i] = pos_angle_preds[mask].mean(dim=0)
        compacted_angle_preds = compacted_angle_preds[bmsk].view(-1, 2, pos_angle_preds.shape[-1])
        compacted_angle_preds = self.angle_coder.decode(compacted_angle_preds, keepdim=False)
        compacted_agnostic_mask = None
        if self.rotation_agnostic_classes:
            compacted_labels = torch.empty(bid.shape, dtype=torch.float, device=bid.device)
            for i in range(len(bid)):
                mask = (idx == i)
                if mask.sum() == 0: 
                    compacted_labels[i] = 0
                else:
                    compacted_labels[i] = pos_labels[mask].float().mean(dim=0)
            compacted_labels = compacted_labels[bmsk].long().view(-1, 2)[:, 0]
            compacted_agnostic_mask = self._get_rotation_agnostic_mask(compacted_labels)

            compacted_angle_preds[compacted_agnostic_mask] = compacted_angle_targets[compacted_agnostic_mask]

        bid_vaild = bid[bmsk]

        return bid_vaild, b_flp, compacted_angle_preds, compacted_angle_targets, compacted_agnostic_mask

    def forward_train(self,
                      stage,
                      x,
                      img_metas,
                      gt_points_all, 
                      proposal_list_base, 
                      proposals_list,     
                      proposals_valid_list,
                      neg_proposal_list,
                      neg_weight_list,
                      gt_true_bboxes,
                      gt_labels,
                      dynamic_weight,
                      use_dview = False,
                      gt_points_ignore=None,
                      gt_masks=None,
                      ):
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            if not use_dview:
                if self.use_angle_loss and stage==0: 
                    self.use_snap_loss = True
                    angle_preds = self.angle_forward(x)
                    featmap_sizes = [featmap.size()[-2:] for featmap in angle_preds]
                    dtype = angle_preds[0].dtype
                    ## 样本分配
                    all_level_points = self.grid_priors(featmap_sizes, dtype=dtype, device=x[0].device)
                    labels, angle_targets, bid_targets = self.get_targets(
                        x, all_level_points, gt_points_all, proposal_list_base, gt_labels, img_metas)
                    num_imgs = angle_preds[0].size(0)
                    flatten_angle_preds = [
                        angle_pred.permute(0, 2, 3, 
                                        1).reshape(-1, self.angle_coder.encode_size)
                        for angle_pred in angle_preds
                    ]
                    flatten_angle_preds = torch.cat(flatten_angle_preds)
                    flatten_labels = torch.cat(labels)
                    flatten_angle_targets = torch.cat(angle_targets)
                    flatten_bid_targets = torch.cat(bid_targets)
                    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
                    bg_class_ind = self.num_classes
                    pos_inds = ((flatten_labels >= 0) &
                                (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
                    num_pos = torch.tensor(
                        len(pos_inds), dtype=dtype, device=angle_preds[0].device)
                    num_pos = max(reduce_mean(num_pos), num_pos.new_tensor(1.0))
                    pos_labels = flatten_labels[pos_inds]
                    
                    # pos sample for each lvl
                    pos_inds_perlvl = []
                    pos_angle_preds_perlvl = []
                    pos_bid_targets_perlvl = []
                    pos_angle_targets_perlvl = []
                    pos_labels_perlvl = []
                    start_idx = 0
                    for lvl, label_per_lvl in enumerate(labels):
                        end_idx = start_idx + len(label_per_lvl)
                        # pos_ind_perlvl = pos_inds[(pos_inds>start_idx) & (pos_inds<end_idx)]
                        pos_ind_perlvl = ((label_per_lvl >= 0) &
                                        (label_per_lvl < bg_class_ind)).nonzero().reshape(-1)
                        pos_ind_perlvl = pos_ind_perlvl + start_idx
                        pos_inds_perlvl.append(pos_ind_perlvl)
                        pos_bid_targets_perlvl.append(flatten_bid_targets[pos_ind_perlvl])
                        pos_angle_preds_perlvl.append(flatten_angle_preds[pos_ind_perlvl])
                        pos_angle_targets_perlvl.append(flatten_angle_targets[pos_ind_perlvl])
                        pos_labels_perlvl.append(flatten_labels[pos_ind_perlvl])
                        start_idx = end_idx

                    pos_angle_preds = torch.cat(pos_angle_preds_perlvl)
                    pos_angle_targets = torch.cat(pos_angle_targets_perlvl)
                    pos_bid_targets = torch.cat(pos_bid_targets_perlvl)

                    if len(pos_inds) > 0:
                        if self.rotation_agnostic_classes:
                            pos_agnostic_mask = self._get_rotation_agnostic_mask(
                                pos_labels)
                            target_mask = torch.abs(
                                pos_angle_targets[pos_agnostic_mask]) < math.pi / 4
                            if target_mask.size(0)>0:
                                pos_angle_targets[pos_agnostic_mask] = torch.where(
                                    target_mask, 0., -math.pi / 2)
                        # Self-Supervision
                        # Aggregate targets of the same bbox based on their identical bid
                        bid_vaild, b_flp, compacted_angle_preds, compacted_angle_targets, \
                        compacted_agnostic_mask = self.GetCompacted(dtype, pos_angle_preds,
                                                                    pos_angle_targets, pos_bid_targets, pos_labels)
                        if b_flp: 
                            d_ang = compacted_angle_preds[:, 0] + compacted_angle_preds[:, 1]
                        else:
                            d_ang = (compacted_angle_preds[:, 0] - compacted_angle_preds[:, 1]) - (compacted_angle_targets[:, 0] - compacted_angle_targets[:, 1])
                        if self.use_snap_loss:
                            d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
                        if compacted_agnostic_mask is not None:
                            d_ang[compacted_agnostic_mask] = 0
                        if len(d_ang) > 0:
                            loss_symmetry_ss = self.loss_symmetry_ss(d_ang, torch.zeros_like(d_ang))
                        else:
                            loss_symmetry_ss = d_ang.sum()

                        # per lvl
                        bid_lvls = []
                        compacted_angle_preds_lvls = []
                        compacted_angle_targets_lvls = []
                        for pos_angle_preds_lvl, pos_angle_targets_lvl, pos_bid_targets_lvl, pos_labels_lvl in zip(pos_angle_preds_perlvl, pos_angle_targets_perlvl, pos_bid_targets_perlvl, pos_labels_perlvl):
                            bid_vaild_lvl, _, compacted_angle_preds_lvl, compacted_angle_targets_lvl, \
                            _ = self.GetCompacted(dtype, pos_angle_preds_lvl,
                                                    pos_angle_targets_lvl, pos_bid_targets_lvl, pos_labels_lvl)
                            #  bid_lvl, idx_lvl = torch.unique(pos_bid_targets_lvl, return_inverse=True)
                            bid_lvls.append(bid_vaild_lvl.long())
                            compacted_angle_preds_lvls.append(compacted_angle_preds_lvl)
                            compacted_angle_targets_lvls.append(compacted_angle_targets_lvl)
                        
                    else:
                        loss_symmetry_ss = pos_angle_preds.sum()
                    losses.update(loss_symmetry_ss = loss_symmetry_ss)
                    
                    flatten_gt_points_all = torch.cat(gt_points_all, dim=0)
                    num_gt_all = flatten_gt_points_all.size(0)

                    #  add angle to proposal
                    if self.add_angle_pred_begin and len(pos_inds) > 0:
                        assert self.use_angle_loss == True
                        # get pos samples via bid
                        num_levels = len(angle_preds)
                        angle_all_lvls = angle_preds
                        num_imgs = angle_preds[0].size(0)

                        # get lvl of proposal
                        proposals_xy_lvls = []
                        for proposals in proposals_list:
                            scale = torch.sqrt(proposals[:, 2] * proposals[:, 3])
                            target_lvls = torch.floor(torch.log2(scale / self.bbox_roi_extractor.finest_scale + 1e-6))
                            target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long().unsqueeze(1)
                            proposals_xy_lvl = torch.cat([proposals[:,0:2], target_lvls], dim=1)
                            proposals_xy_lvls.append(proposals_xy_lvl)

                        # match angle
                        flatten_proposals_xy_lvls = torch.cat(proposals_xy_lvls, dim=0)
                        num_proposals_all = len(flatten_proposals_xy_lvls)
                        num_proposals_bs_list = [len(proposals_xy_lvl) for proposals_xy_lvl in proposals_xy_lvls]
                        empty_angle_add_proposals = torch.zeros((flatten_proposals_xy_lvls.size(0), 2), device=angle_preds[0].device,
                                                            dtype = dtype)
                        empty_angle_add_proposals = empty_angle_add_proposals.reshape(num_gt_all, -1, 2)  # [num_gt, num_pros]
                        flatten_proposals_xy_lvls = flatten_proposals_xy_lvls.reshape(num_gt_all, -1, flatten_proposals_xy_lvls.size(-1))
                        half_gt_num = num_gt_all // 2

                        for gt in range(num_gt_all):
                            view = 'ori' if ((gt + 1) <= half_gt_num ) else 'rot'
                            for lvl in range(num_levels):
                                mask_in_lvl = flatten_proposals_xy_lvls[gt][:,2].long() == lvl
                                vaild_num = mask_in_lvl.sum().item()
                                if vaild_num == 0:
                                    continue
                                mask_in_bid = (bid_lvls[lvl] == (gt + 1 - half_gt_num)) if view == 'rot' \
                                    else (bid_lvls[lvl] == (gt + 1))  # find assigned gt
                                if mask_in_bid.sum()>0:
                                    if view == 'ori':
                                        angle_preds_gt_lvl = compacted_angle_preds_lvls[lvl].reshape(-1)[mask_in_bid][0]
                                    elif view == 'rot':
                                        angle_preds_gt_lvl = compacted_angle_preds_lvls[lvl].reshape(-1)[mask_in_bid][1]
                                    empty_angle_add_proposals[gt][mask_in_lvl,0] = 1
                                    empty_angle_add_proposals[gt][mask_in_lvl,1] = angle_preds_gt_lvl.repeat(vaild_num)

                        empty_angle_add_proposals = empty_angle_add_proposals.reshape(num_proposals_all, empty_angle_add_proposals.size(-1))
                        angle_add_proposals_list = []  # stroage corresponding angle
                        start_idx = 0
                        for num_per_img in num_proposals_bs_list:
                            end_idx = start_idx + num_per_img
                            angle_add_proposals_list.append(empty_angle_add_proposals[start_idx:end_idx,:])
                            start_idx = end_idx
                        ## add angle_pred to all proposals
                        for i, proposals_xy_lvl in enumerate(proposals_xy_lvls):
                            proposals_add_angle = angle_add_proposals_list[i]
                            assigned_ = proposals_add_angle[:, 0] == 1
                            angle_assign = proposals_add_angle[assigned_,1]
                            proposals_list[i][assigned_, -1] = angle_assign.detach()
                            # proposals_list[i][assigned_, -1] += angle_assign.detach()
                            # proposals_list[i][assigned_, -1] += angle_assign
                            unassigned_mask = ~assigned_
                            if unassigned_mask.sum() == 0:
                                continue
                            
                            # # assign left proposals
                            for lvl in range(num_levels):
                                mask_in_lvl = proposals_xy_lvl[:, 2].long() == lvl
                                if mask_in_lvl.sum() == 0:
                                    continue
                                grids_xy_lvl = all_level_points[lvl]
                                proposals_xy_unassign = proposals_xy_lvl[mask_in_lvl&unassigned_mask][:, 0:2]
                                if proposals_xy_unassign.size(0) > 0:
                                    ct_xy = proposals_xy_unassign[:, None, :]
                                    points_gt_dist = torch.norm(grids_xy_lvl - ct_xy, dim=2)
                                    min_dist_index = torch.argmin(points_gt_dist, dim=1)
                                    angle_lvl = angle_all_lvls[lvl].permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
                                    angle_preds_lvl = self.angle_coder.decode(angle_lvl, keepdim=False).reshape(num_imgs, -1)
                                    angle_pred = angle_preds_lvl[i][min_dist_index].squeeze().detach()
                                    proposals_list[i][mask_in_lvl & unassigned_mask, -1] = angle_pred
            
            bbox_results = self._bbox_forward_train(x, proposal_list_base, proposals_list, proposals_valid_list,
                                                    neg_proposal_list,
                                                    neg_weight_list,
                                                    gt_true_bboxes,
                                                    gt_labels, dynamic_weight,
                                                    img_metas, stage)

            losses.update(bbox_results['loss_instance_mil'])
            
        return losses, bbox_results, bbox_results['pseudo_boxes'], bbox_results['dynamic_weight']

    def _bbox_forward_train(self, x, proposal_list_base, proposals_list, proposals_valid_list, neg_proposal_list,
                            neg_weight_list, gt_true_bboxes,
                            gt_labels,
                            cascade_weight,
                            img_metas, stage):
        """Run forward function and calculate loss for box head in training."""

        rois = rbbox2roi(proposals_list)  # [bs,cx,cy,w,h,a]
        bbox_results = self._bbox_forward(x, rois, gt_true_bboxes, stage)
        # bbox_results:{cls_score, ins_score, bbox_pred=reg_box, bbox_feats, num_instance=num_gt}
        gt_labels = torch.cat(gt_labels)
        proposals_valid_list = torch.cat(proposals_valid_list).reshape(
            *bbox_results['cls_score'].shape[:2], 1)

        if neg_proposal_list is not None:
            neg_rois = rbbox2roi(neg_proposal_list)
            neg_bbox_results = self._bbox_forward(x, neg_rois, None, stage)
            neg_cls_scores = neg_bbox_results['cls_score']
            neg_weights = torch.cat(neg_weight_list)
        else:
            neg_cls_scores = None
            neg_weights = None
        reg_box = bbox_results['bbox_pred']
        if reg_box is not None:
            boxes_pred = self.bbox_head.bbox_coder.decode(torch.cat(proposals_list).reshape(-1, 5),
                                                          reg_box.reshape(-1, 5)).reshape(reg_box.shape)
        else:
            boxes_pred = None

        proposals_list_to_merge = proposals_list
        pseudo_boxes, mean_ious, filtered_boxes, filtered_scores, dynamic_weight = self.merge_rbox(bbox_results,
                                                                                                  proposals_list_to_merge,
                                                                                                  proposals_valid_list,
                                                                                                  gt_labels,
                                                                                                  gt_true_bboxes,
                                                                                                  img_metas, stage)
        bbox_results.update(pseudo_boxes=pseudo_boxes)
        bbox_results.update(dynamic_weight=dynamic_weight.sum(dim=-1))
        bbox_results.update(dynamic_weight_all=dynamic_weight)

        pseudo_boxes = torch.cat(pseudo_boxes)
        if stage == self.num_stages - 1:
            retrain_weights = None ##TO
        else:
            retrain_weights = None
        loss_instance_mil = self.bbox_head.loss_mil(stage, bbox_results['cls_score'], bbox_results['ins_score'],
                                                    proposals_valid_list,
                                                    neg_cls_scores, neg_weights,
                                                    boxes_pred, gt_labels,
                                                    torch.cat(proposal_list_base),
                                                    torch.cat(gt_true_bboxes),
                                                    label_weights=cascade_weight,
                                                    retrain_weights=retrain_weights)
        loss_instance_mil.update({"mean_ious": mean_ious[-1]})
        loss_instance_mil.update({"s": mean_ious[0]})
        loss_instance_mil.update({"m": mean_ious[1]})
        loss_instance_mil.update({"l": mean_ious[2]})
        loss_instance_mil.update({"h": mean_ious[3]})
        bbox_results.update(loss_instance_mil=loss_instance_mil)
        return bbox_results
    
    def _bbox_forward(self, x, rois, gt_true_bboxes, stage):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, ins_score, reg_box = self.bbox_head(bbox_feats, stage)
        # positive sample
        if gt_true_bboxes is not None:
            num_gt = torch.cat(gt_true_bboxes).shape[0]
            assert num_gt != 0, f'num_gt = 0 {gt_true_bboxes}'

            cls_score = cls_score.view(num_gt, -1, cls_score.shape[-1])  # (num_gt, num_proposals, num_cls+1)
            ins_score = ins_score.view(num_gt, -1, ins_score.shape[-1])  # (num_gt, num_proposals, num_cls+1)

            if reg_box is not None:
                reg_box = reg_box.view(num_gt, -1, reg_box.shape[-1])

            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=num_gt)
            return bbox_results
        # megative sample
        else:
            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=None)
            return bbox_results

    def _bbox_forward_test(self, x, rois, gt_true_bboxes, stage):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        # print(f'roi num:{len(rois)}')
        
        max_roi_num = 10000
        if len(rois) > max_roi_num: # too many rois-OOM
            cls_score_list = []
            ins_score_list = []
            reg_box_list = []

            iter = len(rois)//max_roi_num
            for i in range(iter):
                rois_tmp = rois[max_roi_num*i:max_roi_num*(i+1)]
                bbox_feats = self.bbox_roi_extractor(
                    x[:self.bbox_roi_extractor.num_inputs], rois_tmp)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)
                cls_score_tmp, ins_score_tmp, reg_box_tmp = self.bbox_head(bbox_feats, stage)
                cls_score_list.append(cls_score_tmp)
                ins_score_list.append(ins_score_tmp)
                reg_box_list.append(reg_box_tmp)

                del rois_tmp
                del bbox_feats
                del cls_score_tmp
                del ins_score_tmp
                del reg_box_tmp

            rois_last = rois[iter*max_roi_num:]
            bbox_feats = self.bbox_roi_extractor(
                    x[:self.bbox_roi_extractor.num_inputs], rois_last)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score_tmp, ins_score_tmp, reg_box_tmp = self.bbox_head(bbox_feats, stage)
            cls_score_list.append(cls_score_tmp)
            ins_score_list.append(ins_score_tmp)
            reg_box_list.append(reg_box_tmp)
        
            cls_score = torch.cat(cls_score_list)
            ins_score = torch.cat(ins_score_list)
            if reg_box_tmp is not None:
                reg_box = torch.cat(reg_box_list)
            else:
                reg_box = None

            del rois_last
            del bbox_feats
            del cls_score_tmp
            del ins_score_tmp
            del reg_box_tmp
            bbox_feats = None

        else:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, ins_score, reg_box = self.bbox_head(bbox_feats, stage)

        # positive sample
        if gt_true_bboxes is not None:
            num_gt = torch.cat(gt_true_bboxes).shape[0]
            assert num_gt != 0, f'num_gt = 0 {gt_true_bboxes}'

            cls_score = cls_score.view(num_gt, -1, cls_score.shape[-1])
            ins_score = ins_score.view(num_gt, -1, ins_score.shape[-1])
            if reg_box is not None:
                reg_box = reg_box.view(num_gt, -1, reg_box.shape[-1])

            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=num_gt)
            return bbox_results
        else:
            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=None)
            return bbox_results

    def merge_rbox_single(self, cls_score, ins_score, dynamic_weight, gt_label, proposals, img_metas, stage):
        '''
        proposals: Tensor[n, 5], [cx, cy, w, h, a]
        '''
        if stage < self.num_stages - 1:
            merge_mode = 'weighted_clsins_topk'
        elif stage == self.num_stages - 1:
            merge_mode = 'weighted_clsins_topk'

        proposals = proposals.reshape(cls_score.shape[0], cls_score.shape[1], 5)
        h, w, c = img_metas['img_shape']
        num_gt, num_gen = proposals.shape[:2]
        if merge_mode == 'weighted_cls_topk':
            cls_score_, idx = cls_score.topk(k=self.topk2, dim=1)
            weight = cls_score_.unsqueeze(2).repeat([1, 1, 5])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8) 
            boxes = (proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx] * weight).sum(dim=1)
            return boxes, None, None

        if merge_mode == 'weighted_clsins_topk':
            if stage == 0:
                k = self.topk1
            else:
                k = self.topk2
            dynamic_weight_, idx = dynamic_weight.topk(k=k, dim=1) 
            weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 5]) 
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)  
            filtered_boxes = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
            boxes = (filtered_boxes * weight).sum(dim=1)
            h, w, _ = img_metas['img_shape']
            boxes[:, 0:4:2] = boxes[:, 0:4:2].clamp(0, w)
            boxes[:, 1:4:2] = boxes[:, 1:4:2].clamp(0, h)
            filtered_scores = dict(cls_score=cls_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                   ins_score=ins_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                   dynamic_weight=dynamic_weight_)

            return boxes, filtered_boxes, filtered_scores

    def merge_rbox(self, bbox_results, proposals_list, proposals_valid_list, gt_labels, gt_bboxes, img_metas, stage):
        cls_scores = bbox_results['cls_score']
        ins_scores = bbox_results['ins_score']
        num_instances = bbox_results['num_instance']
        # num_gt = len(gt_labels)
        # num_gt * num_box * num_class
        if stage < 1:
            cls_scores = cls_scores.softmax(dim=-1)
        else:
            cls_scores = cls_scores.sigmoid()
        ins_scores = ins_scores.softmax(dim=-2) * proposals_valid_list
        ins_scores = F.normalize(ins_scores, dim=1, p=1)
        cls_scores = cls_scores * proposals_valid_list

        dynamic_weight = (cls_scores * ins_scores)
        dynamic_weight = dynamic_weight[torch.arange(len(cls_scores)), :, gt_labels]
        cls_scores = cls_scores[torch.arange(len(cls_scores)), :, gt_labels]
        ins_scores = ins_scores[torch.arange(len(cls_scores)), :, gt_labels]

        # split batch
        batch_gt = [len(b) for b in gt_bboxes]
        cls_scores = torch.split(cls_scores, batch_gt)
        ins_scores = torch.split(ins_scores, batch_gt)
        gt_labels = torch.split(gt_labels, batch_gt)
        dynamic_weight_list = torch.split(dynamic_weight, batch_gt)
        if not isinstance(proposals_list, list):
            proposals_list = torch.split(proposals_list, batch_gt)
        stage_ = [stage for _ in range(len(cls_scores))]
        boxes, filtered_boxes, filtered_scores = multi_apply(self.merge_rbox_single, cls_scores, ins_scores,
                                                             dynamic_weight_list,
                                                             gt_labels,
                                                             proposals_list,
                                                             img_metas, stage_)
        pseudo_boxes = torch.cat(boxes).detach()
        half_num = pseudo_boxes.size(0)//2
        half_num_gt = len(gt_bboxes)//2
        if cls_scores[0].requires_grad:  # training
            if self.use_angle_loss and self.add_angle_pred_begin:
                iou1 = box_iou_rotated(pseudo_boxes, torch.cat(gt_bboxes), aligned=True)
                gt_xywh = torch.cat(gt_bboxes)[:,0:4]
                if pseudo_boxes[:half_num,:].size(0)==torch.cat(gt_bboxes[:half_num_gt]).size(0):
                    if self.not_use_rot_mil:
                        iou1 = box_iou_rotated(pseudo_boxes[:half_num,:], torch.cat(gt_bboxes[:half_num_gt]), aligned=True)
                        gt_xywh = torch.cat(gt_bboxes[:half_num_gt])[:,0:4]
            else:
                gt_hbox = torch.cat(gt_bboxes)
                gt_hbox = obb2cxcywh_le90(gt_hbox)
                iou1 = box_iou_rotated(pseudo_boxes, gt_hbox, aligned=True) 
                gt_xywh = gt_hbox[:,0:4]
                if self.not_use_rot_mil:
                    iou1 = box_iou_rotated(pseudo_boxes[:half_num,:], gt_hbox[:half_num,:], aligned=True)
                    gt_xywh = gt_hbox[:half_num,0:4]
            
        else: # evaluation
            # print('----evalutaion-----')
            gt_xywh = torch.cat(gt_bboxes)[:,0:4]
            if self.use_angle_loss:
                if self.agnostic_resize_classes and stage>0:
                    labels_ = torch.cat(gt_labels)
                    for id in self.agnostic_resize_classes:
                        pseudo_boxes[labels_ == id, 2:4] *= 0.65
                iou1 = box_iou_rotated(pseudo_boxes, torch.cat(gt_bboxes), aligned=True)
            else:
                # print('----evalutaion no add angle!-----')
                gt_hbox = torch.cat(gt_bboxes)
                gt_hbox = obb2cxcywh_le90(gt_hbox) 
                iou1 = box_iou_rotated(pseudo_boxes, gt_hbox, aligned=True)
                
        ### scale mean iou
        scale = gt_xywh[:, 2] * gt_xywh[:, 3]
        mean_iou_s = iou1[scale < 32 ** 2].sum() / (len(iou1[scale < 32 ** 2]) + 1e-5)
        mean_iou_m = iou1[(scale > 32 ** 2) * (scale < 64 ** 2)].sum() / (len(
            iou1[(scale > 32 ** 2) * (scale < 64 ** 2)]) + 1e-5)
        mean_iou_l = iou1[(scale > 64 ** 2) * (scale < 128 ** 2)].sum() / (len(
            iou1[(scale > 64 ** 2) * (scale < 128 ** 2)]) + 1e-5)
        mean_iou_h = iou1[scale > 128 ** 2].sum() / (len(iou1[scale > 128 ** 2]) + 1e-5)

        mean_ious_all = iou1.mean()
        mean_ious = [mean_iou_s, mean_iou_m, mean_iou_l, mean_iou_h, mean_ious_all]

        if self.test_mean_iou and stage == 1:
            self.sum_iou += iou1.sum()
            self.sum_num += len(iou1)
            print('\r', self.sum_iou / self.sum_num, end='', flush=True)

        pseudo_boxes = torch.split(pseudo_boxes, batch_gt)
        return list(pseudo_boxes), mean_ious, list(filtered_boxes), list(filtered_scores), dynamic_weight.detach()

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results
    
    @torch.no_grad()
    def simple_test(self,
                    stage,
                    x,
                    proposal_list,
                    proposals_valid_list,
                    gt_points,
                    gt_true_bboxes,
                    gt_labels,
                    gt_anns_id,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.use_angle_loss:
            if self.add_angle_pred_begin and stage==0:
                ## get 获取dense angle_pred
                angle_preds = self.angle_forward(x)  
                featmap_sizes = [featmap.size()[-2:] for featmap in angle_preds]

                num_imgs = angle_preds[0].size(0)
                batch_gt = [len(b) for b in gt_points]
                dtype = angle_preds[0].dtype
                num_levels = len(angle_preds)
                # grids
                all_level_points = self.grid_priors(featmap_sizes, dtype=dtype, device=x[0].device)
                angle_pred_lvls = []
                for angle_pred in angle_preds:
                    angle_lvl = angle_pred.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
                    angle_pred_lvl = self.angle_coder.decode(angle_lvl, keepdim=False).reshape(num_imgs, -1)
                    angle_pred_lvls.append(angle_pred_lvl)
                gt_preds_all_lvl = []
                for lvl, grids_xy_lvl in enumerate(all_level_points):
                    angle_pred_list = []
                    for img, gt_point in enumerate(gt_points):
                        points_gt_dist = torch.norm(grids_xy_lvl.unsqueeze(0) - gt_point.unsqueeze(1), dim=2)
                        min_dist_index = torch.argmin(points_gt_dist, dim=1)
                        angle_pred = angle_pred_lvls[lvl][img,min_dist_index]
                        angle_pred_list.append(angle_pred)
                    gt_preds_all_lvl.append(angle_pred_list)

                for img, proposal in enumerate(proposal_list):
                    proposal_per_gt = proposal.reshape(batch_gt[img],-1,5)
                    proposal_per_gt_angle_add = torch.zeros((batch_gt[img],proposal_per_gt.size(1)),
                                                            device=proposal.device)
                    for gt in range(len(proposal_per_gt)):
                        ###
                        scale = torch.sqrt(proposal_per_gt[gt][:, 2] * proposal_per_gt[gt][:, 3])
                        target_lvls = torch.floor(torch.log2(scale / self.bbox_roi_extractor.finest_scale + 1e-6))
                        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
                        # angle_add = gt_preds_all_lvl[target_lvls][img][gt]
                        angle_add = torch.stack([gt_preds_all_lvl[target_lvls[i]][img][gt]
                                                for i in range(len(target_lvls))])
                        proposal_per_gt_angle_add[gt] = angle_add

                    proposal_list[img][:,-1] += proposal_per_gt_angle_add.reshape(-1)

        det_bboxes, det_labels, pseudo_bboxes = self.simple_test_bboxes(
            x, img_metas, proposal_list, proposals_valid_list, gt_true_bboxes, gt_labels, gt_anns_id, stage, self.test_cfg,
            rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results, pseudo_bboxes

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           proposals_valid_list,
                           gt_true_bboxes,
                           gt_labels,
                           gt_anns_id,
                           stage,
                           rcnn_test_cfg,
                           rescale=False):
        
        # get origin input shape to support onnx dynamic input shape
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if self.rotation_agnostic_classes:
            for pr in range(len(proposals)):
            # for bboxes, labels in zip(proposals, gt_labels):
                bboxes = proposals[pr]
                labels = gt_labels[pr]
                bboxes = bboxes.reshape(labels.shape[0],-1,5)
                for id in self.rotation_agnostic_classes:
                    bboxes[labels == id,:,-1] = 0
                bboxes = bboxes.reshape(-1,5)
                proposals[pr] = bboxes
                    
        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward_test(x, rois, gt_true_bboxes, stage)
        proposals_valid_list = torch.cat(proposals_valid_list).reshape(
            *bbox_results['cls_score'].shape[:2], 1)

        pseudo_boxes, mean_ious, filtered_boxes, filtered_scores, dynamic_weight = self.merge_rbox(bbox_results,
                                                                                                  proposals,
                                                                                                  proposals_valid_list,
                                                                                                  torch.cat(gt_labels),
                                                                                                  gt_true_bboxes,
                                                                                                  img_metas, stage)
        pseudo_boxes_out = copy.deepcopy(pseudo_boxes)

        det_bboxes, det_labels = self.pseudobox_to_result(pseudo_boxes, gt_labels, dynamic_weight, gt_anns_id,
                                                          scale_factors, rescale)
        
        return det_bboxes, det_labels, pseudo_boxes_out

    def pseudobox_to_result(self, pseudo_boxes, gt_labels, dynamic_weight, gt_anns_id, scale_factors, rescale):
        det_bboxes = []
        det_labels = []
        batch_gt = [len(b) for b in gt_labels]
        dynamic_weight = torch.split(dynamic_weight, batch_gt)
        for i in range(len(pseudo_boxes)):
            boxes = pseudo_boxes[i]
            labels = gt_labels[i]

            if rescale and boxes.shape[0] > 0:
                scale_factor = boxes.new_tensor(scale_factors[i]).unsqueeze(0).repeat(
                    1, boxes.size(-1) // 4)
                boxes[:,0:4] /= scale_factor

            boxes = torch.cat([boxes, dynamic_weight[i].sum(dim=1, keepdim=True)], dim=1)
            gt_anns_id_single = gt_anns_id[i]
            boxes = torch.cat([boxes, gt_anns_id_single.unsqueeze(1)], dim=1)
            det_bboxes.append(boxes)
            det_labels.append(labels)
        return det_bboxes, det_labels

    def test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.test_bboxes(x, img_metas,
                                                  proposal_list,
                                                  self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if self.with_mask:
            segm_results = self.test_mask(x, img_metas, det_bboxes,
                                          det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
            -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals
        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels

