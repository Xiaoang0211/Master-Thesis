import argparse
import time

import torch.optim as optim

import sys
sys.path.append('.')

from modules.trainers.epoch_base_trainer import EpochBasedTrainer
from modules.datasets.loaders import get_train_val_data_loader
from modules.trainers.losses_node_match import *
from modules.trainers.loss_graph_match import *
import modules.networks.pointnetvlad as PNV
import loss_pointnetvlad as PNV_loss
from configs_node_match import config, update_config

class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, parser=None):
        super().__init__(cfg, parser)
        
        # Model Specific params
        self.modules = cfg.modules
        
        # Loss params for node embeddings
        self.cfg = cfg
        self.zoom = cfg.loss.zoom
        self.weight_align_loss = cfg.loss.alignment_loss_weight
        self.weight_contrastive_loss = cfg.loss.constrastive_loss_weight
        
        # dataloader
        start_time = time.time()
        train_loader, val_loader = get_train_val_data_loader(cfg)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model: sgaligner + sg net
        model = self.create_model()
        self.register_model(model)
        
        # train: training parameters
        self.num_pos = cfg.train.num_pos
        self.num_neg = cfg.train.num_neg
        self.batch_size = cfg.train.batch_size

        # loss function and params
        self.loss_retrieval = PNV_loss.triplet_loss_wrapper
        self.parameters = filter(lambda p: p.requires_grad, model.parameters())
        
        # optimizer and scheduler
        optimizer_PNV = optim.Adam(self.parameters, lr=cfg.optim.lr_node, weight_decay=cfg.optim.weight_decay)

        self.register_optimizer_PNV(optimizer_PNV)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        # self.register_scheduler(scheduler)

        self.logger.info('Initialisation Complete')

    def create_model(self):
        model = PNV.PointNetVlad(global_feat=True, feature_transform=True,
                             max_pool=False, output_dim=256, num_points=4096)
    
        message = 'Model created'
        self.logger.info(message)
        return model

    def train_step(self, epoch, iteration, data_dict_batched): # processing of 1 query
        # for data_dict in data_dict_batched:
        output_dict = self.model(data_dict_batched)
        target_scores = output_dict['targets']
        pred_scores = output_dict['predictions']
        final_loss = {}
        loss_dict_retrieval = self.loss_retrieval(pred_scores, target_scores)
        final_loss["loss"] = loss_dict_retrieval
        return output_dict, final_loss

    def val_step(self, epoch, iteration, data_dict_batched):
        output_dict = self.model(data_dict_batched)
        target_scores = output_dict['targets']
        pred_scores = output_dict['predictions']
        final_loss = {}
        loss_dict_retrieval = self.loss_retrieval(pred_scores, target_scores)
        final_loss["loss"] = loss_dict_retrieval
        return output_dict, final_loss

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        # self.loss_retrieval.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        self.training = True
        self.model.train()
        # self.loss_retrieval.train()
        torch.set_grad_enabled(True)

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--snapshot', default=None, help='load from snapshot')
    parser.add_argument('--epoch', type=int, default=None, help='load epoch')
    parser.add_argument('--log_steps', type=int, default=500, help='logging steps')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for ddp')

    args = parser.parse_args()
    return parser, args

def main():
    parser, args = parse_args()
    cfg = update_config(config, args.config)
    trainer = Trainer(cfg, parser)
    trainer.run()

if __name__ == '__main__':
    main()