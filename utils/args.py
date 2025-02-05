import argparse
from functools import partial
from pathlib import Path

from models.backbones import BACKBONES
from models.backbones.resnet_fx import ResNetFX
from models.bb_regressors import REGRESSORS
from models.disnet import DisNet
from models.distformer import DistFormer
from models.naive_mlp import MLP
from models.svr import SVR
from models.temporal_compression import TEMPORAL_COMPRESSORS
from models.zhu import ZHU

DEFAULT_DS_STATS = {"h_mean": 118.69,
                    "w_mean": 50.21,
                    "d_mean": 29.71,
                    "h_std": 112.60,
                    "w_std": 58.77,
                    "d_std": 18.99,
                    "d_95_percentile": 71.58}

MODELS = {'mlp': MLP,
          'rfx': ResNetFX,
          'distformer': DistFormer,
          'zhu': partial(ZHU, enhanced=False),
          'zhu_enhanced': partial(ZHU, enhanced=True),
          'disnet': DisNet,
          'svr': SVR}


def check_positive(strictly_positive: bool = True):
    def check_positive_inner(value):
        try:
            ivalue = int(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"{value} is not an integer") from e
        if strictly_positive:
            if ivalue <= 0:
                raise argparse.ArgumentTypeError(
                    f"{value} is an invalid strictly positive int value")
        else:
            if ivalue < 0:
                raise argparse.ArgumentTypeError(
                    f"{value} is an invalid positive int value")
        return ivalue

    return check_positive_inner


def set_default_args(args) -> argparse.Namespace:
    if isinstance(MODELS[args.model], partial):
        temporal = MODELS[args.model].func.TEMPORAL
    else:
        temporal = MODELS[args.model].TEMPORAL

    args.clip_len = args.clip_len if temporal else 1

    if 'fpn' in args.backbone and args.fpn_idx_lstm != 'none':
        args.temporal_compression = 'none'
    if args.model == 'zhu':
        args.fpn_bottleneck_lstm = False
        args.fpn_idx_lstm = 'none'
    return args


def parse_args(default: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--log_each_step", default=True, action="store_true", )
    parser.add_argument("--no_log_each_step", dest="log_each_step", action="store_false", )
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument('--wandb_tag', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from exp_log_path')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--save_nothing', action='store_true', default=False)

    # PATHS
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--ds_path', type=str, default='data/motsynth')
    parser.add_argument('--ds_did3md', type=str, default=None)
    parser.add_argument('--annotations_path', type=str)
    parser.add_argument('--detections_path', type=str, default='tracking/yolox_bounding_boxes/sequences',
                        help='Directory where to load detections')
    parser.add_argument('--mot_challenge_anns_path', type=str, default=None)
    parser.add_argument('--mot_challenge_path', type=str, default=None)
    parser.add_argument('--exp_log_path', type=str, default=None)

    # MODEL
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr_gamma', type=float, default=1, help='Multiplicative factor of backbone lr')
    parser.add_argument('--change_lr', action='store_true', default=False)
    parser.add_argument('--optimizer', type=str, default='adam', choices=('adam', 'sgd', 'adam8bit'))
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=('cosine', 'plateau', 'none'))
    parser.add_argument('--lr_gamma', type=float, default=1,
                        help='Multiplicative factor of learning rate decay each --lr_decay_steps. (disabled if 1)')
    parser.add_argument('--lr_decay_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--max_patience', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--accumulation_steps', type=check_positive(), default=4)
    parser.add_argument('--model', type=str, choices=MODELS.keys(), default='distformer')
    parser.add_argument('--loss', type=str, default='gaussian', choices=("l1", "mse", "laplacian", "gaussian", "balanced_l1"))
    parser.add_argument('--loss_warmup', type=check_positive(strictly_positive=False), default=0)
    parser.add_argument('--loss_warmup_start', type=check_positive(strictly_positive=False), default=0)
    parser.add_argument('--nearness', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--fpn_hdim', type=int, default=256)
    parser.add_argument('--fpn_idx_lstm', type=str, default='none',
                        choices=('none', 'c3', 'c4', 'c5', 'p3', 'p4', 'p5'))
    parser.add_argument('--estimator_input', type=int, default='1024')
    parser.add_argument('--transformer_dropout', type=float, default=0.)
    parser.add_argument('--transformer_aggregate', type=int, default=1)
    parser.add_argument('--use_geom', action='store_true', default=False)
    parser.add_argument('--roi_op', type=str, default='align', choices=('align', 'pool'))

    # Masked AE
    parser.add_argument('--mae_patch_size', type=int, default=1)
    parser.add_argument('--mae_detach', action='store_true', default=True)
    parser.add_argument('--no_mae_detach', dest='mae_detach', action='store_false')
    parser.add_argument('--mae_masking', type=float, default=0.5)
    parser.add_argument('--mae_alpha', type=float, default=1., help='Weight for MAE loss')
    parser.add_argument('--mae_warmup', type=check_positive(strictly_positive=False), default=0)
    parser.add_argument('--mae_target', type=str, default='input', choices=('input', 'latent'))
    parser.add_argument('--mae_crop_size', type=int, nargs='+', default=(128, 64), help='(image_crop_h, image_crop_w)')
    parser.add_argument('--norm_pix_loss', action='store_true', default=False)
    parser.add_argument('--save_freq', type=int, default=2000)
    parser.add_argument('--mae_loss_only', action='store_true', default=False)
    parser.add_argument('--mae_encoder_layers', type=int, default=6)
    parser.add_argument('--mae_decoder_layers', type=int, default=2)
    parser.add_argument('--mae_activation', type=str, default='relu', choices=('relu', 'gelu'))
    parser.add_argument('--mae_vit_pretrained', action='store_true', default=True)
    parser.add_argument('--mae_version', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--mae_loss', type=str, default='mse', choices=('mse', 'focal'))
    parser.add_argument('--mae_loss_no_masked', type=int, default=0, help='Use the recon loss also for non-masked patches')
    parser.add_argument('--mae_alpha_loss_no_masked', type=float, default=.1)
    parser.add_argument('--mae_min_brightness', type=float, default=0, help='Minimum brightness of the image')
    parser.add_argument('--mae_min_bbox_h', type=int, default=0)
    parser.add_argument('--mae_min_bbox_w', type=int, default=0)
    parser.add_argument('--mae_encoder_dim', type=int, default=768)
    parser.add_argument('--mae_weight_night', type=float, default=0)
    parser.add_argument('--mae_weight_small_bboxes', type=float, default=0)
    parser.add_argument('--mae_global_encoder', type=str, default='self_attention', choices=('none', 'self_attention', 'gcn', 'gat'))
    parser.add_argument('--mae_eval_masking', action='store_true', default=False)
    parser.add_argument('--mae_eval_masking_ratio', type=float, default=0.5)
    parser.add_argument('--disable_mae_epoch', type=int, default=-1, help='Disable MAE loss after this epoch (-1 is always enabled)')
    parser.add_argument('--classify_anchor', type=int, default=0, choices=(0, 1), help='Use the class token to classify the anchor')

    # Backbone
    parser.add_argument('--backbone', type=str, default='resnetfpn34', choices=BACKBONES.keys())
    parser.add_argument('--shallow', action='store_true', default=False, help='Stop at Layer 3 (only Resnet)')
    parser.add_argument("--pretrain", default="imagenet", type=str, choices=("none", "imagenet", "imagenet22k"))
    parser.add_argument('--use_centers', action='store_true', default=True)
    parser.add_argument('--no_use_centers', dest='use_centers', action='store_false')
    parser.add_argument('--residual_centers', action='store_true', default=False)
    parser.add_argument('--bam_centers', action='store_true', default=False,
                        help='Gives the centers also in input to the BAM backbone')

    # Temporal Compression
    parser.add_argument("--temporal_compression", type=str,
                        default="identity", choices=TEMPORAL_COMPRESSORS.keys())

    # Regressor
    parser.add_argument('--regressor', type=str, default='roi_pooling', choices=REGRESSORS.keys())
    parser.add_argument('--pool_size', type=int, nargs='+', default=(8, 8))
    parser.add_argument('--adjacency', type=str, default='iou', choices=('distance', 'iou'),
                        help='Adjacency matrix type (only for ROI GCN')
    parser.add_argument('--alpha_zhu', type=float, default=1e-2, help='Zhu loss weight')
    parser.add_argument('--avg_pool_post_roi', type=int, default=0, help='Average pool after ROI pooling (0 disables it)')

    # Graph smoothing
    parser.add_argument('--smoothing', action='store_true', default=False, help='Use graph smoothing')
    parser.add_argument('--alpha_smooth', type=float, default=1e-2, help='smoothing loss weight')
    parser.add_argument('--threshold', type=float, default=1.0, help='threshold for adjacency in graph smoothing')
    parser.add_argument('--normalize_laplacian', action='store_true', default=False)
    parser.add_argument('--sample_from_gaussian', action='store_true', default=False)

    # Non-Local Block
    parser.add_argument('--phi_ksize', type=int, default=1, choices=(1, 3), help='[NLB] Kernel size for phi')
    parser.add_argument('--batch_norm_nlb', action='store_true', default=True, help='[NLB] Use batch norm in NLB')
    parser.add_argument('--no_batch_norm_nlb', dest='batch_norm_nlb', action='store_false')
    parser.add_argument('--sub_sample_nlb', action='store_true', default=False, help='[NLB] Use sub-sampling in NLB')
    parser.add_argument('--nlb_mode', type=str, default='gaussian', choices=('gaussian', 'dot', 'embedded'),
                        help='[NLB] Mode for NLB')

    # Inference
    parser.add_argument('--test_only', action='store_true', default=False, help='Test only without training')
    parser.add_argument('--infer_only', action='store_true', default=False, help='Infer on test set only (save detections)')
    parser.add_argument('--infer_gt', action='store_true', default=False, help='Infer on the ground truth bboxes')
    parser.add_argument('--infer_num_chunks', type=int, default=1, help='Number of chunks to split the test set')
    parser.add_argument('--infer_chunk_idx', type=int, help='Index of the chunk to infer on')
    parser.add_argument('--infer_frame_features', type=int, choices=(0, 1), help='Infer on frame features', default=1)
    parser.add_argument('--infer_mot_version', type=str, choices=('17', '20'), help='MOT version to infer on')
    parser.add_argument('--infer_split', type=str, choices=('train', 'test'), help='Split to infer on')
    parser.add_argument('--infer_sequence', type=str, help='Infer only on this sequence name')
    parser.add_argument('--test_all', action='store_true', default=False, help='Test on the entire test set')
    parser.add_argument('--show_bev', action='store_true', default=False, help='Show BEV')
    parser.add_argument('--bev_out_path', type=str, default=None, help='Path to save BEV images')
    # DATASET
    parser.add_argument('--dataset', type=str, default='motsynth',
                        choices=('kitti', 'kitti_3D', 'kitti_didm3d', 'motsynth', 'kitti_tracking', 'nuscenes'))
    parser.add_argument('--classes_to_keep', type=str, nargs='+', default=('all'), choices=('all', 'car', 'pedestrian', 'cyclist'))
    parser.add_argument('--load_ds_into_ram', action='store_true', default=False)
    parser.add_argument('--use_debug_dataset', action='store_true', default=False)
    parser.add_argument('--augmentation', type=str, default='torchvision', choices=('torchvision'))
    parser.add_argument('--test_sampling_stride', type=check_positive(), default=400)
    parser.add_argument('--train_sampling_stride', type=check_positive(), default=50)
    parser.add_argument('--input_h_w', type=int, nargs='+', default=(720, 1280))
    parser.add_argument('--min_visibility', type=float, default=0.18)  # 0.18 \approx 4/22
    parser.add_argument('--crop_range', type=float, nargs='+', default=(0.0, 0.25))
    parser.add_argument('--crop_mode', type=str, default='random', choices=('random', 'center'))
    parser.add_argument('--clip_len', type=int, default=1)
    parser.add_argument('--max_stride', type=int, default=8)
    parser.add_argument('--stride_sampling', type=str, default='fixed', choices=('fixed', 'normal', 'uniform'))
    parser.add_argument('--sampling', type=str, default='naive', choices=('naive', 'smart'))
    parser.add_argument('--scale_factor', type=float, default=0.5)
    parser.add_argument('--use_heads', action='store_true', default=False)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--ds_stats', type=dict, default=DEFAULT_DS_STATS)
    parser.add_argument('--augment_centers', action='store_true', default=False)
    parser.add_argument('--augment_centers_std', type=float, default=1)
    parser.add_argument('--aug_gaussian_blur_kernel', type=int, default=7)
    parser.add_argument('--aug_gaussian_blur_sigma', nargs='+', type=float, default=(2.0, 4.0))
    parser.add_argument('--noisy_bb', action='store_true')
    parser.add_argument('--noisy_bb_test', action='store_true')
    parser.add_argument('--noisy_bb_with_iou', action='store_true')
    parser.add_argument('--noisy_bb_iou_th', type=float, default=0.5)
    parser.add_argument('--long_range', action='store_true', default=True, help='Eval KITTI on long range distances')
    parser.add_argument('--max_dist_test', type=float, default=0, help='Max distance to consider for test (<0 is all)')

    # MONOLOCO
    parser.add_argument('--monoloco_dropout', type=float, default=0.2)
    parser.add_argument('--monoloco_linear_size', type=int, default=256)
    parser.add_argument('--monoloco_activation', type=str,
                        default='relu', choices=('relu', 'leaky_relu', 'silu'))
    parser.add_argument('--monoloco_batchnorm',
                        action='store_true', help='Whether to use batch_norm')
    parser.add_argument('--simple_pose', action='store_true',
                        help='Use simple pose (14 joints)')
    parser.add_argument('--occluded_ds', action='store_true',
                        default=False, help='Use occluded dataset')

    # PERFORMANCE
    parser.add_argument('--use_deepspeed',
                        action='store_true', help='Use deepspeed')
    parser.add_argument('--performace_save_path', type=str, default=None)

    return parser.parse_args([] if default else None)
