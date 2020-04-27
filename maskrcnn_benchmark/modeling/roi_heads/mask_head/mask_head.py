# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator, make_cascadercnn_roi_mask_loss_evaluator1, make_cascadercnn_roi_mask_loss_evaluator2, make_cascadercnn_roi_mask_loss_evaluator3


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)


class CASCADEROIMaskHead1(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(CASCADEROIMaskHead1, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor1 = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor1 = make_roi_mask_predictor(
            cfg, self.feature_extractor1.out_channels)
        self.post_processor1 = make_roi_mask_post_processor(cfg)
        self.loss_evaluator1 = make_cascadercnn_roi_mask_loss_evaluator1(cfg)
        self.loss_weight1 = cfg.MODEL.CASCADE_RCNN_ROI_HEADS.LOSS_WEIGHT1

    def forward(self, features, proposals1, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals1 = proposals1
            proposals1, positive_inds1 = keep_only_positive_boxes(proposals1)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x1 = features
            x1 = x1[torch.cat(positive_inds1, dim=0)]
        else:
            x1 = self.feature_extractor1(features, proposals1)
        mask_logits1 = self.predictor1(x1)

        if not self.training:
            result1 = self.post_processor1(mask_logits1, proposals1)
            return x1, result1, {}

        loss_mask1 = self.loss_evaluator1(proposals1, mask_logits1, targets)

        return x1, all_proposals1, dict(loss_mask1=self.loss_weight1*loss_mask1)


def build_cascadercnn_roi_mask_head1(cfg, in_channels):
    return CASCADEROIMaskHead1(cfg, in_channels)


class CASCADEROIMaskHead2(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(CASCADEROIMaskHead2, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor2 = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor2 = make_roi_mask_predictor(
            cfg, self.feature_extractor2.out_channels)
        self.post_processor2 = make_roi_mask_post_processor(cfg)
        self.loss_evaluator2 = make_cascadercnn_roi_mask_loss_evaluator2(cfg)
        self.loss_weight2 = cfg.MODEL.CASCADE_RCNN_ROI_HEADS.LOSS_WEIGHT2

    def forward(self, features, proposals2, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals2 = proposals2
            proposals2, positive_inds2 = keep_only_positive_boxes(proposals2)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x2 = features
            x2 = x2[torch.cat(positive_inds2, dim=0)]
        else:
            x2 = self.feature_extractor2(features, proposals2)
        mask_logits2 = self.predictor2(x2)

        if not self.training:
            result2 = self.post_processor2(mask_logits2, proposals2)
            return x2, result2, {}

        loss_mask2 = self.loss_evaluator2(proposals2, mask_logits2, targets)

        return x2, all_proposals2, dict(loss_mask2=self.loss_weight2*loss_mask2)


def build_cascadercnn_roi_mask_head2(cfg, in_channels):
    return CASCADEROIMaskHead2(cfg, in_channels)


class CASCADEROIMaskHead3(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(CASCADEROIMaskHead3, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor3 = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor3 = make_roi_mask_predictor(
            cfg, self.feature_extractor3.out_channels)
        self.post_processor3 = make_roi_mask_post_processor(cfg)
        self.loss_evaluator3 = make_cascadercnn_roi_mask_loss_evaluator3(cfg)
        self.loss_weight3 = cfg.MODEL.CASCADE_RCNN_ROI_HEADS.LOSS_WEIGHT3

    def forward(self, features, proposals3, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals3 = proposals3
            proposals3, positive_inds3 = keep_only_positive_boxes(proposals3)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x3 = features
            x3 = x3[torch.cat(positive_inds3, dim=0)]
        else:
            x3 = self.feature_extractor3(features, proposals3)
        mask_logits3 = self.predictor3(x3)

        if not self.training:
            result3 = self.post_processor3(mask_logits3, proposals3)
            return x3, result3, {}

        loss_mask3 = self.loss_evaluator3(proposals3, mask_logits3, targets)

        return x3, all_proposals3, dict(loss_mask3=self.loss_weight3*loss_mask3)


def build_cascadercnn_roi_mask_head3(cfg, in_channels):
    return CASCADEROIMaskHead3(cfg, in_channels)
