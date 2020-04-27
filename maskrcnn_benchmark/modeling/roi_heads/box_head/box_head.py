# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor, make_cascadercnn_roi_box_post_processor1, make_cascadercnn_roi_box_post_processor2, make_cascadercnn_roi_box_post_processor2
from .loss import make_roi_box_loss_evaluator, make_cascadercnn_roi_box_loss_evaluator1, make_cascadercnn_roi_box_loss_evaluator2, make_cascadercnn_roi_box_loss_evaluator3


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)


class CascadeRCNNROIBoxHead1(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(CascadeRCNNROIBoxHead1, self).__init__()
        self.feature_extractor1 = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor1 = make_roi_box_predictor(
            cfg, self.feature_extractor1.out_channels)
        self.post_processor1 = make_cascadercnn_roi_box_post_processor1(cfg)
        self.loss_evaluator1 = make_cascadercnn_roi_box_loss_evaluator1(cfg)
        self.loss_weight1 = cfg.MODEL.CASCADE_RCNN_ROI_HEADS.LOSS_WEIGHT1

    def forward(self, features, proposals1, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals1 = self.loss_evaluator1.subsample(proposals1, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x1 = self.feature_extractor1(features, proposals1)
        # final classifier that converts the features into predictions
        class_logits1, box_regression1 = self.predictor1(x1)

        if not self.training:
            result1 = self.post_processor1((class_logits1, box_regression1), proposals1)
            return x1, result1, {}

        loss_classifier1, loss_box_reg1 = self.loss_evaluator1(
            [class_logits1], [box_regression1]
        )
        return (
            x1,
            proposals1,
            dict(loss_classifier1=self.loss_weight1*loss_classifier1, loss_box_reg1=self.loss_weight1*loss_box_reg1),
        )


def build_cascadercnn_roi_box_head1(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return CascadeRCNNROIBoxHead1(cfg, in_channels)


class CascadeRCNNROIBoxHead2(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(CascadeRCNNROIBoxHead2, self).__init__()
        self.feature_extractor2 = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor2 = make_roi_box_predictor(
            cfg, self.feature_extractor2.out_channels)
        self.post_processor2 = make_cascadercnn_roi_box_post_processor2(cfg)
        self.loss_evaluator2 = make_cascadercnn_roi_box_loss_evaluator2(cfg)
        self.loss_weight2 = cfg.MODEL.CASCADE_RCNN_ROI_HEADS.LOSS_WEIGHT2

    def forward(self, features, proposals2, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals2 = self.loss_evaluator2.subsample(proposals2, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x2 = self.feature_extractor2(features, proposals2)
        # final classifier that converts the features into predictions
        class_logits2, box_regression2 = self.predictor2(x2)

        if not self.training:
            result2 = self.post_processor2((class_logits2, box_regression2), proposals2)
            return x2, result2, {}

        loss_classifier2, loss_box_reg2 = self.loss_evaluator2(
            [class_logits2], [box_regression2]
        )
        return (
            x2,
            proposals2,
            dict(loss_classifier2=self.loss_weight2*loss_classifier2, loss_box_reg2=self.loss_weight2*loss_box_reg2),
        )


def build_cascadercnn_roi_box_head2(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return CascadeRCNNROIBoxHead2(cfg, in_channels)


class CascadeRCNNROIBoxHead3(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(CascadeRCNNROIBoxHead3, self).__init__()
        self.feature_extractor3 = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor3 = make_roi_box_predictor(
            cfg, self.feature_extractor3.out_channels)
        self.post_processor3 = make_cascadercnn_roi_box_post_processor3(cfg)
        self.loss_evaluator3 = make_cascadercnn_roi_box_loss_evaluator3(cfg)
        self.loss_weight3 = cfg.MODEL.CASCADE_RCNN_ROI_HEADS.LOSS_WEIGHT3

    def forward(self, features, proposals3, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals3 = self.loss_evaluator3.subsample(proposals3, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x3 = self.feature_extractor3(features, proposals3)
        # final classifier that converts the features into predictions
        class_logits3, box_regression3 = self.predictor3(x3)

        if not self.training:
            result3 = self.post_processor3((class_logits3, box_regression3), proposals3)
            return x3, result3, {}

        loss_classifier3, loss_box_reg3 = self.loss_evaluator3(
            [class_logits3], [box_regression3]
        )
        return (
            x3,
            proposals3,
            dict(loss_classifier3=self.loss_weight3*loss_classifier3, loss_box_reg3=self.loss_weight3*loss_box_reg3),
        )


def build_cascadercnn_roi_box_head3(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return CascadeRCNNROIBoxHead3(cfg, in_channels)
