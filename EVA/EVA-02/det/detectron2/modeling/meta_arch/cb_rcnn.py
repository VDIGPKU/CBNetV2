# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import warnings

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

import logging
logger = logging.getLogger(__name__)

from .rcnn import GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class CBGeneralizedRCNN(GeneralizedRCNN):

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features, cb_features = self.backbone(images.tensor)

        torch.cuda.empty_cache()

        losses = {}
        loss = self.inner_forward(images, features, gt_instances, batched_inputs)
        # losses.update(loss)
        for k, v in loss.items():
            new_k = 'first_' + k
            losses[new_k] = v * 0.5

        torch.cuda.empty_cache()

        cb_loss = self.inner_forward(images, cb_features, gt_instances, batched_inputs)
        for k, v in cb_loss.items():
            new_k = 'second_' + k
            losses[new_k] = v
        
        torch.cuda.empty_cache()

        return losses

    def inner_forward(self, images, features, gt_instances, batched_inputs):

        for k, v in features.items():
            # feat_isnan = torch.isnan(v).any()
            # feat_isinf = torch.isinf(v).any()
            # if feat_isnan or feat_isinf:
            #     logger.info("============ feat_isnan={}, feat_isinf={} ===============".format(feat_isnan, feat_isinf))
            features[k] = v.float()

        with torch.cuda.amp.autocast(enabled=False):
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        keep_all_before_merge: bool = False,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
            keep_all_before_merge

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        _, features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None, keep_all_before_merge=keep_all_before_merge)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            # optionally update score using maskness
            if self.roi_heads.maskness_thresh is not None:
                for pred_inst in results:
                    # pred_inst._fields.keys():  dict_keys(['pred_boxes', 'scores', 'pred_classes', 'pred_masks'])
                    pred_masks = pred_inst.pred_masks  # (num_inst, 1, 28, 28)
                    scores = pred_inst.scores  # (num_inst, )
                    # sigmoid already applied
                    binary_masks = pred_masks > self.roi_heads.maskness_thresh
                    seg_scores = (pred_masks * binary_masks.float()).sum((1, 2, 3)) / binary_masks.sum((1, 2, 3))
                    seg_scores[binary_masks.sum((1, 2, 3)) == 0] = 0  # avoid nan
                    updated_scores = scores * seg_scores
                    pred_inst.set('scores', updated_scores)
                    # update order
                    scores, indices = updated_scores.sort(descending=True)
                    pred_inst = pred_inst[indices]
                    assert (pred_inst.scores == scores).all()

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    