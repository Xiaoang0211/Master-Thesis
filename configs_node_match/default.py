from yacs.config import CfgNode as CN
import os.path as osp

from utilities.utils_node_match import common

_C = CN()

# common 
_C.seed = 42
_C.num_workers = 4
_C.model_name = ''
_C.modules = []
_C.registration = True
_C.modality = ''
_C.use_predicted = False
_C.scan_type = 'subscan'

# path params
_C.data = CN()
_C.data.name = 'Scan3R'
_C.data.root_dir = ''
_C.data.label_file_name = ''
_C.data.ply_subfix = ''
_C.data.seg_subfix = ''
_C.data.aggre_subfix = ''
_C.data.pred_subfix = ''
_C.data.rand_mod_rate = 0

# choosing subset of dataset
_C.lower_bound = 0.1
_C.upper_bound = 1.0
_C.w_or_wo_rescans = 'w'

# preprocess params
_C.preprocess = CN()
_C.preprocess.pc_resolutions = [512] # [32, 64, 128, 256, 512]
_C.preprocess.subscenes_per_scene = 7
_C.preprocess.min_obj_points = 50
_C.preprocess.anchor_type_name = ''
_C.preprocess.label_type = 'Scannet20'
_C.preprocess.search_method = 'BBOX'
_C.preprocess.radius_receptive = 0.5
_C.preprocess.max_distance = 0.1
_C.preprocess.filter_segment_size = 512
_C.preprocess.filter_corr_thres = 0.5
_C.preprocess.filter_occ_ratio = 0.75
_C.preprocess.name_same_segment = 'same part'

# Training params
_C.train = CN()
_C.train.batch_size = 1
_C.train.pc_res = 512
_C.train.use_augmentation = True
_C.train.rot_factor = 1.0
_C.train.augmentation_noise = 0.005
_C.train.freeze_sgaligner = True
_C.train.num_pos = 2
_C.train.num_neg = 20

# Validation params
_C.val = CN()
_C.val.data_mode = 'orig'
_C.val.batch_size = 1
_C.val.pc_res = 512
_C.val.overlap_low = 0.0
_C.val.overlap_high = 0.0

# Test params
_C.test = CN()
_C.test.data_mode = 'orig'
_C.test.batch_size = 1
_C.test.pc_res = 512
_C.test.overlap_low = 0.0
_C.test.overlap_high = 0.0

# model param
_C.model = CN()
_C.model.modal_emb_dim = 128 # default 100
_C.model.rel_dim = 41
_C.model.attr_dim = 164
_C.model.alignment_thresh = 0.4

# optim
_C.optim = CN()
_C.optim.lr_node = 1e-6
_C.optim.lr_graph = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 20
_C.optim.grad_acc_steps = 1

# loss
_C.loss = CN()
_C.loss.alignment_loss_weight = 1.0
_C.loss.constrastive_loss_weight = 1.0
_C.loss.zoom = 0.1

# registration model params
_C.reg_model = CN()
_C.reg_model.K = 2
_C.reg_model.neighbor_limits = [38, 36, 36, 38]
_C.reg_model.num_p2p_corrs = 20000
_C.reg_model.corr_score_thresh = 0.1
_C.reg_model.rmse_thresh = 0.2 
_C.reg_model.inlier_ratio_thresh = 0.05
_C.reg_model.ransac_threshold = 0.03
_C.reg_model.ransac_min_iters = 5000
_C.reg_model.ransac_max_iters = 5000
_C.reg_model.ransac_use_sprt = False

# inference
_C.metrics = CN()
_C.metrics.all_k = [1, 2, 3, 4, 5]

# args for sg net
_C.args_sg = CN()
_C.args_sg.cuda = 0
_C.args_sg.filters_1 = 256 #256
_C.args_sg.filters_2 = 256 #256
_C.args_sg.filters_3 = 128 #128 
_C.args_sg.feat_dim_in = 100 # default 100 if without linear projection then 256
_C.args_sg.tensor_neurons = 64
_C.args_sg.bottle_neck_neurons = 32
_C.args_sg.K = 10
_C.args_sg.lr = 0.0001
_C.args_sg.weight_decay = 0.0005
_C.args_sg.gpu = 0
_C.args_sg.num_head = 4
_C.args_sg.logdir = "./logs_k10"
_C.args_sg.node_num = 30
_C.args_sg.eval_output = "./eva"
_C.args_sg.show = True



def update_config(cfg, filename, ensure_dir=True):
    cfg.defrost()
    cfg.merge_from_file(filename)
    
    if ensure_dir:
        cfg.working_dir = osp.dirname(osp.abspath(__file__))
        cfg.root_dir = osp.dirname(_C.working_dir)
        cfg.exp_name = '_'.join(_C.modules)
        cfg.output_dir = osp.join(_C.root_dir, 'output', _C.data.name, _C.model_name, _C.exp_name)
        # cfg.output_dir = cfg.output_dir + '_nonlinear_after_aggregat'
        # cfg.output_dir = cfg.output_dir + '_larger_batch_16_2_20'
        cfg.output_dir = cfg.output_dir + '_MLP_first_one_channel_for_dgcnn'
        # cfg.output_dir = cfg.output_dir + '_dgcnn_ntn_wo_cattention'
        # cfg.output_dir = cfg.output_dir + '_direct_aggregat_wo_cattention_cosine' # the real baseline
        # cfg.output_dir = cfg.output_dir + '_direct_aggregat_ntn_cattention'
        cfg.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
        cfg.log_dir = osp.join(_C.output_dir, 'logs')
        cfg.event_dir = osp.join(_C.output_dir, 'events')
        cfg.registration = False
        
        common.ensure_dir(cfg.output_dir)
        common.ensure_dir(cfg.snapshot_dir)
        common.ensure_dir(cfg.log_dir)
        common.ensure_dir(cfg.event_dir)
    
    cfg.freeze()
    
    return cfg