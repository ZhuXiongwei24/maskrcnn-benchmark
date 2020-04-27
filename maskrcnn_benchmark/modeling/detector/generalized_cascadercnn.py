# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_cascadercnn_roi_heads1, build_cascadercnn_roi_heads2, build_cascadercnn_roi_heads3


class GeneralizedCascadeRCNN(nn.Module):
    """
    Main class for Generalized Cascade R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedCascadeRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads1 = build_cascadercnn_roi_heads1(cfg, self.backbone.out_channels)
        self.roi_heads2 = build_cascadercnn_roi_heads2(cfg, self.backbone.out_channels)
        self.roi_heads3 = build_cascadercnn_roi_heads3(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads1:
            x1, result1, detector_losses1 = self.roi_heads1(features, proposals, targets)
        if self.roi_heads2:
            x2, result2, detector_losses2 = self.roi_heads2(features, result1, targets)
        if self.roi_heads3:
            x3, result3, detector_losses3 = self.roi_heads3(features, result2, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses1)
            losses.update(detector_losses2)
            losses.update(detector_losses3)
            losses.update(proposal_losses)
            return losses

        return result2
