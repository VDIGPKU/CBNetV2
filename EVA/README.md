# CBNet: A Composite Backbone Network Architecture for Object Detection

### EVA02 (1536x1536)
|  Backbone  | Lr Schd | mask mAP (test-dev) | #params |                            config                            |                       model                       |
| :--------: | :-----: | :-----------------: | :-----: | :----------------------------------------------------------: | :-----------------------------------------------: |
| DB-EVA02-L |   1x    |        56.1         |  674M   | [config](EVA-02/det/projects/ViTDet/configs/eva2_o365_to_coco/cb_eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py) | [HF](https://huggingface.co/weeewe/CBNetV2-EVA02) |

- **Pre-trained models of EVA02 can be downloaded from [EVA02 pretrain](https://github.com/baaivision/EVA/tree/master/EVA-02/det)**.

## Usage

Please refer to [EVA](https://github.com/baaivision/EVA/tree/master/EVA-02/) for code clone, installation, and dataset preparation.
Then, replace or add the files from EVA-02 in this CBNet repository to original EVA repository.

Download pretraining weight (eva02_L_m38m_to_o365.pth) from [EVA02 pretrain](https://github.com/baaivision/EVA/tree/master/EVA-02/det).
Then, run
```
python get_cb_ckpt.py
```
to create CBNet pretraining weight (cb_eva02_L_m38m_to_o365.pth).

### Training

To train CBNet-EVA-L with pre-trained models, run:
```
# multi-gpu training
python tools/lazyconfig_train_net.py \
    --num-gpus N_GPU  \
    --config-file projects/ViTDet/configs/eva2_o365_to_coco/cb_eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py \
    train.output_dir=YOUR_OUTPUT_PATH \
    train.init_checkpoint=PATH_TO/cb_eva02_L_m38m_to_o365.pth

### Inference on Test

To train CBNet-EVA-L with pre-trained models, run:
python tools/lazyconfig_train_net.py \
 --num-gpus N_GPU \
 --config-file projects/ViTDet/configs/eva2_o365_to_coco/cb_eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py \
 --eval-only \
 train.model_ema.use_ema_weights_for_eval_only=True \
 model.roi_heads.use_soft_nms=True \
 model.roi_heads.class_wise=True \
 model.roi_heads.method=linear \
 model.roi_heads.iou_threshold=0.5 \
 model.roi_heads.override_score_thresh=0.0 \
 model.roi_heads.maskness_thresh=0.5 \
 train.init_checkpoint=YOUR_OUTPUT_PATH/model_final.pth \
 dataloader.evaluator.output_dir=YOUR_OUTPUT_PATH
```


Another  example, to train a Mask R-CNN model with a `Duel-Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

## Citation
If you use our code/model, please consider to cite our paper [CBNet: A Composite Backbone Network Architecture for Object Detection](http://arxiv.org/abs/2107.00420).
```
@ARTICLE{9932281,
  author={Liang, Tingting and Chu, Xiaojie and Liu, Yudong and Wang, Yongtao and Tang, Zhi and Chu, Wei and Chen, Jingdong and Ling, Haibin},
  journal={IEEE Transactions on Image Processing}, 
  title={CBNet: A Composite Backbone Network Architecture for Object Detection}, 
  year={2022},
  volume={31},
  pages={6893-6906},
  doi={10.1109/TIP.2022.3216771}}
```

## License
The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.


## Other Links
> **Original CBNet**: See [CBNet: A Novel Composite Backbone Network Architecture for Object Detection](https://github.com/VDIGPKU/CBNet).