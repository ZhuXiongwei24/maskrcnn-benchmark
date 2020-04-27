# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head, build_cascadercnn_roi_box_head1, build_cascadercnn_roi_box_head2, build_cascadercnn_roi_box_head3
from .mask_head.mask_head import build_roi_mask_head, build_cascadercnn_roi_mask_head1, build_cascadercnn_roi_mask_head2, build_cascadercnn_roi_mask_head3
from .keypoint_head.keypoint_head import build_roi_keypoint_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads


class CombinedCascadeRCNNROIHeads1(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedCascadeRCNNROIHeads1, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor1 = self.box1.feature_extractor1

    def forward(self, features, proposals, targets=None):
        losses1 = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x1, detections1, loss_box1 = self.box1(features, proposals, targets)
        losses1.update(loss_box1)
        if self.cfg.MODEL.MASK_ON:
            mask_features1 = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features1 = x1
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x1, detections1, loss_mask1 = self.mask1(mask_features1, detections1, targets)
            losses1.update(loss_mask1)
        return x1, detections1, losses1


def build_cascadercnn_roi_heads1(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    cascadercnn_roi_heads1 = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        cascadercnn_roi_heads1.append(("box1", build_cascadercnn_roi_box_head1(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        cascadercnn_roi_heads1.append(("mask1", build_cascadercnn_roi_mask_head1(cfg, in_channels)))

    # combine individual heads in a single module
    if cascadercnn_roi_heads1:
        cascadercnn_roi_heads1 = CombinedCascadeRCNNROIHeads1(cfg, cascadercnn_roi_heads1)

    return cascadercnn_roi_heads1


class CombinedCascadeRCNNROIHeads2(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedCascadeRCNNROIHeads2, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor2 = self.box2.feature_extractor2

    def forward(self, features, proposals, targets=None):
        losses2 = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x2, detections2, loss_box2 = self.box2(features, proposals, targets)
        losses2.update(loss_box2)
        if self.cfg.MODEL.MASK_ON:
            mask_features2 = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features2 = x2
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x2, detections2, loss_mask2 = self.mask2(mask_features2, detections2, targets)
            losses2.update(loss_mask2)
        return x2, detections2, losses2


def build_cascadercnn_roi_heads2(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    cascadercnn_roi_heads2 = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        cascadercnn_roi_heads2.append(("box2", build_cascadercnn_roi_box_head2(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        cascadercnn_roi_heads2.append(("mask2", build_cascadercnn_roi_mask_head2(cfg, in_channels)))

    # combine individual heads in a single module
    if cascadercnn_roi_heads2:
        cascadercnn_roi_heads2 = CombinedCascadeRCNNROIHeads2(cfg, cascadercnn_roi_heads2)

    return cascadercnn_roi_heads2


class CombinedCascadeRCNNROIHeads3(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedCascadeRCNNROIHeads3, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor3 = self.box3.feature_extractor3

    def forward(self, features, proposals, targets=None):
        losses3 = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x3, detections3, loss_box3 = self.box3(features, proposals, targets)
        losses3.update(loss_box3)
        if self.cfg.MODEL.MASK_ON:
            mask_features3 = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features3 = x3
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x3, detections3, loss_mask3 = self.mask3(mask_features3, detections3, targets)
            losses3.update(loss_mask3)
        return x3, detections3, losses3


def build_cascadercnn_roi_heads3(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    cascadercnn_roi_heads3 = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        cascadercnn_roi_heads3.append(("box3", build_cascadercnn_roi_box_head3(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        cascadercnn_roi_heads3.append(("mask3", build_cascadercnn_roi_mask_head3(cfg, in_channels)))

    # combine individual heads in a single module
    if cascadercnn_roi_heads3:
        cascadercnn_roi_heads3 = CombinedCascadeRCNNROIHeads3(cfg, cascadercnn_roi_heads3)

    return cascadercnn_roi_heads3

