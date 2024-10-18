import logging
import math
from functools import partial

import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous

from .backbone import Backbone
from .utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
    VisionRotaryEmbeddingFast,
)

try:
    import xformers.ops as xops
except:
    pass

try:
    from apex.normalization import FusedLayerNorm
except:
    pass


logger = logging.getLogger(__name__)
from mmcv.cnn import constant_init

from .vit import ViT, SimpleFeaturePyramid
__all__ = ["CBViT", "CBSimpleFeaturePyramid"]



class CBViT(ViT):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        cb_out_index,
        del_patch_embed=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cb_out_index = cb_out_index
        if del_patch_embed:
            del self.patch_embed
            del self.pos_embed
            self.patch_embed = None
            self.pos_embed = None
        
        assert cb_out_index[-1]+1 == len(self.blocks)

    def forward(self, x, cb_feats=None):
        assert cb_feats is not None or self.patch_embed is not None
        if self.patch_embed is None:
            x = cb_feats[0] + cb_feats[1]
            # print(len(cb_feats))
        else:
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + get_abs_pos(
                    self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
                )
        
        cb_output = [x]
        j = 2
        # print(len(cb_feats))
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.cb_out_index:
                cb_output.append(x)
                if cb_feats is not None and i != self.cb_out_index[-1]:
                    x = x + cb_feats[j]
                    j += 1
                    # print(i, j)
        # assert j == len(self.cb_out_index) + 1
        # print(j, len(self.cb_out_index))

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        
        return outputs, cb_output


class CBSimpleFeaturePyramid(SimpleFeaturePyramid):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net,
        cb_net,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        super(SimpleFeaturePyramid, self).__init__()
        assert isinstance(net, Backbone)

        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.cb_net = cb_net

        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

        self.cb_linears = nn.ModuleList()
        for i in range(4):
            linears = nn.ModuleList()
            jrange = 4 - i
            for j in range(jrange):
                linears.append(nn.Conv2d(net.embed_dim, net.embed_dim, 1))
            self.cb_linears.append(linears)
        
        self.init_cb_weights()
    
    def init_cb_weights(self):
        for ls in self.cb_linears:
            for m in ls:
                # if isinstance(m, nn.Sequential):
                #     constant_init(m[-1], 0)
                #     # torch.nn.init.constant(m, val)
                # else:
                constant_init(m, 0)
    
    def forward_neck(self, bottom_up_features):
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features, cb_output = self.net(x)
        results = self.forward_neck(bottom_up_features)

        cb_feats = [cb_output[0]]
        for i in range(4):
            feats = 0
            jrange = 4 - i
            for j in range(jrange):
                feats += self.cb_linears[i][j](cb_output[i+1].permute(0, 3, 1, 2))
            cb_feats.append(feats.permute(0, 2, 3, 1))

        cb_bottom_up_features, _ = self.cb_net(x, cb_feats)
        cb_results = self.forward_neck(cb_bottom_up_features)

        return [results, cb_results]



        


