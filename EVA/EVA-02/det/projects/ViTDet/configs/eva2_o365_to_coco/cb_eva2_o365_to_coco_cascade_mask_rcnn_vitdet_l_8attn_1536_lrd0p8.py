from functools import partial

from ..common.coco_loader_lsj_1536 import dataloader
from .cb_cascade_mask_rcnn_vitdet_b_100ep import (
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

from detectron2.config import LazyCall as L
from fvcore.common.param_scheduler import *
from detectron2.solver import WarmupParamScheduler



model.backbone.net.img_size = 1536  
model.backbone.square_pad = 1536  
model.backbone.net.patch_size = 16  
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.3

model.backbone.net.cb_out_index = [2, 5, 20, 23]
model.backbone.net.del_patch_embed = False

# 2, 5, 8, 11, 14, 17, 20, 23 for global attention
# 2 2 18 2 swin-L
# 2 5 20 23
model.backbone.net.window_block_indexes = (
    list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
)

model.backbone.cb_net.img_size = 1536  
model.backbone.cb_net.patch_size = 16  
model.backbone.cb_net.window_size = 16
model.backbone.cb_net.embed_dim = 1024
model.backbone.cb_net.depth = 24
model.backbone.cb_net.num_heads = 16
model.backbone.cb_net.mlp_ratio = 4*2/3
model.backbone.cb_net.use_act_checkpoint = True
model.backbone.cb_net.drop_path_rate = 0.3

model.backbone.cb_net.cb_out_index = [2, 5, 20, 23]
model.backbone.cb_net.del_patch_embed = True

# 2, 5, 8, 11, 14, 17, 20, 23 for global attention
# 2 2 18 2 swin-L
# 2 5 20 23
model.backbone.cb_net.window_block_indexes = (
    list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
)


optimizer.lr=4e-5

optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

train.max_iter = 40000

train.model_ema.enabled=True
train.model_ema.device="cuda"
train.model_ema.decay=0.9999

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1,
        end_value=1,
    ),
    warmup_length=0.01,
    warmup_factor=0.001,
)

dataloader.test.num_workers=0
dataloader.train.total_batch_size=64

dataloader.test.dataset.names = "coco_2017_test-dev"
dataloader.evaluator.output_dir = './cb_output_eva_trainval_results'

train.checkpointer.period=1000
train.checkpointer.max_to_keep=10  # options for PeriodicCheckpointer
train.eval_period=40000


