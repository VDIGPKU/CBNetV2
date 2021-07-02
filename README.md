# CBNetV2: A Novel Composite Backbone Network Architecture for Object Detection

This repo contains the supported code and configuration files to reproduce object detection results of [CBNetV2](http://arxiv.org/abs/2107.00420). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

Contact us with tingtingliang@pku.edu.cn, wyt@pku.edu.cn.

## Results and Models

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DB-Swin-T | ImageNet-1K | 3x | 50.2 | 44.5 | 76M | 357G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | 


### Cascade Mask R-CNN
| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DB-Swin-T | ImageNet-1K | 3x | 53.6 | 46.2 | 114M | 836G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | 

### Cascade Mask R-CNN (1600x1400)
| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DB-Res2Net | ImageNet-1K | 20e | 56.3 | 48.6 | 114M | 836G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | 
| DB-Res2Net | ImageNet-1K | 20e + 1x (swa) | 56.3 | 48.6 | 114M | 836G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | 
| DB-Swin-S | ImageNet-1K | 3x | 56.3 | 48.6 | 114M | 836G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | 

### HTC (1600x1400)
| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DB-Swin-B | ImageNet-22K | 20e | 57.9 | 50.2 | 114M | 836G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | 
| DB-Swin-B | ImageNet-22K | 20e + 1x (swa) | 58.2 | 51.1 | 114M | 836G | - | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | 


**Notes**: 

- **Pre-trained models of Swin Transformer can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer)**.
- Access code for `baidu` is `swin`.

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox
# or (w/ bbox_mask)
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox
# or (w/ bbox_mask)
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bboxsegm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
Following [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), we use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
<!-- If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
``` -->

## Citation
If you use our code/model, please consider to cite our paper [CBNetV2: A Novel Composite Backbone Network Architecture for Object Detection]().
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## License
The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.


## Other Links
> **Original CBNet**: See [CBNet: A Novel Composite Backbone Network Architecture for Object Detection](https://github.com/VDIGPKU/CBNet).

> **SWA**: See [SWA Object Detection](https://github.com/hyz-xmaster/swa_object_detection)
