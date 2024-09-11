import argparse
import time

import torch.optim as optim

import sys
sys.path.append('.')

from modules.trainers.epoch_base_trainer import EpochBasedTrainer
from modules.datasets.loaders import get_train_val_data_loader
from modules.trainers.losses_node_match import *
from modules.trainers.loss_graph_match import *
from modules.networks.PR_Net import PR_Net
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
        loss_func_metadata = {'zoom' : self.zoom, 'wt_align_loss' : self.weight_align_loss, 
                              'wt_contrastive_loss' : self.weight_contrastive_loss, 'modules' : self.modules}
        self.loss_func_node_match = OverallLoss(ial_loss_layer = self.multi_loss_layer_ial, icl_loss_layer=self.multi_loss_layer_icl, device=self.device, metadata=loss_func_metadata)
        self.loss_func_graph_match = GraphMatchLoss('binary_cross_entropy', self.num_pos, self.num_neg, self.batch_size)
        
        if len(self.modules) > 1:
            self.params_node_encoder = [{'params' : list(self.model.node_encoder.parameters()) + list(self.loss_func_node_match.align_multi_loss_layer.parameters()) + list(self.loss_func_node_match.contrastive_multi_loss_layer.parameters())}]
        else:
            self.params_node_encoder = [{'params' : list(self.model.node_encoder.parameters())}]
        
        # graph_parameters = nn.ParameterList()
        self.params_graph_encoder = self.model.graph_encoder.parameters()
        print(self.params_graph_encoder)
        
        # optimizer and scheduler
        optimizer_node_encoder = optim.Adam(self.params_node_encoder, lr=cfg.optim.lr_node, weight_decay=cfg.optim.weight_decay)
        optimizer_graph_encoder = optim.Adam(self.params_graph_encoder, lr=cfg.optim.lr_graph, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer_node_encoder, optimizer_graph_encoder)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        # self.register_scheduler(scheduler)

        self.logger.info('Initialisation Complete')

    def create_model(self):
        model = PR_Net(self.cfg)
        
        # if len(self.modules) > 1:
        self.multi_loss_layer_icl = CustomMultiLossLayer(loss_num=len(self.modules), device=self.device)
        self.multi_loss_layer_ial = CustomMultiLossLayer(loss_num=len(self.modules), device=self.device)

        message = 'Model created'
        self.logger.info(message)
        return model

    def train_step(self, epoch, iteration, data_dict_batched): # processing of 1 query
        # for data_dict in data_dict_batched:
        output_dict = self.model(data_dict_batched)
        target_scores = output_dict['targets']
        pred_scores = output_dict['predictions']
        final_loss = {}
        if not self.cfg.train.freeze_sgaligner:
            loss_dict_node_match = self.loss_func_node_match(output_dict, data_dict_batched)
            loss_dict_graph_match = self.loss_func_graph_match(pred_scores, target_scores)
            final_loss["loss"] = 0.001 * loss_dict_node_match["loss"] + loss_dict_graph_match
            final_loss["node_loss"] = loss_dict_node_match
            final_loss["graph_loss"] = loss_dict_graph_match
            return output_dict, final_loss
        else:
            loss_dict_graph_match = self.loss_func_graph_match(pred_scores, target_scores)
            final_loss["loss"] = loss_dict_graph_match
            return output_dict, final_loss

    def val_step(self, epoch, iteration, data_dict_batched):
        output_dict = self.model(data_dict_batched)
        target_scores = output_dict['targets']
        pred_scores = output_dict['predictions']
        final_loss = {}
        
        if not self.cfg.train.freeze_sgaligner:
            loss_dict_node_match = self.loss_func_node_match(output_dict, data_dict_batched)
            loss_dict_graph_match = self.loss_func_graph_match(pred_scores, target_scores)
            final_loss["loss"] = 0.01 * loss_dict_node_match["loss"] + loss_dict_graph_match
            final_loss["node_loss"] = loss_dict_node_match
            final_loss["graph_loss"] = loss_dict_graph_match
            return output_dict, final_loss
        else:
            loss_dict_graph_match = self.loss_func_graph_match(pred_scores, target_scores)
            final_loss["loss"] = loss_dict_graph_match
            return output_dict, final_loss
        # # node matching loss
        # loss_dict_node_match = self.loss_func_node_match(output_dict_node, data_dict_batched)
        
        # # graph matching loss
        # loss_dict_graph_match = self.loss_func_graph_match(output_scores, target_scores)
        
        # return output_dict_node, loss_dict_node_match, loss_dict_graph_match

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        self.multi_loss_layer_icl.eval()
        self.multi_loss_layer_ial.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        self.training = True
        self.model.train()
        self.multi_loss_layer_ial.train()
        self.multi_loss_layer_icl.train()
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