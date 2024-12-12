import torch
import pytorch_lightning as pl
import numpy as np
from torch_scatter import scatter
from functools import partial
from models.configs import models_dict


class Pl_model_wrapper(pl.LightningModule):
    def __init__(self, 
                 model_name, 
                 cfg,
                device
                ):
        super(Pl_model_wrapper, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        model_class = models_dict[model_name]
        self.model = model_class(**cfg.model[model_name])
        self.lr = cfg.train.lr
        self.weight_decay = cfg.train.weight_decay
        self.ipm_steps = cfg.other.ipm_steps
        
        assert 0. <= cfg.other.ipm_alpha <= 1.
        self.step_weight = torch.tensor([cfg.other.ipm_alpha ** (cfg.other.ipm_steps - l - 1)
                                         for l in range(cfg.other.ipm_steps)],
                                        dtype=torch.float, device=device)[None]
        # self.best_val_loss = 1.e8
        self.best_val_objgap = 100.
        self.best_val_consgap = 100.
        self.patience = 0

        self.loss_target = cfg.loss.loss.split('+')
        self.loss_weight = {'primal': cfg.loss.loss_weight_x,
                                       'objgap': cfg.loss.loss_weight_obj,
                                       'constraint': cfg.loss.loss_weight_cons}
        
        if cfg.loss.losstype == 'l2':
            self.loss_func = partial(torch.pow, exponent=2)
        elif cfg.loss.losstype == 'l1':
            self.loss_func = torch.abs

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, data, batch_idx):
        vals, _ = self(data)
        batch_size = data.batch_size
        loss = self.get_loss(vals, data)
        self.log('train_loss',loss,prog_bar=True, batch_size=batch_size, logger=True)
        return loss

    def validation_step(self, data, batch_idx):
        vals, _ = self(data)
        batch_size = data.batch_size
        cons_gap = torch.abs(self.get_constraint_violation(vals, data))[:, -1].mean()
        obj_gap = torch.abs(self.get_obj_metric(data, vals, hard_non_negative=True))[:, -1].mean()
        self.log('cons_gap', cons_gap, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True)
        self.log('obj_gap', obj_gap, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True)
        return obj_gap

    def test_step(self, data, batch_idx):
        vals, _ = self(data)
        cons_gap = np.abs(self.get_constraint_violation(vals, data).detach().cpu().numpy())
        obj_gap = np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy())
        self.log('cons_gap_test', cons_gap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('obj_gap_test', obj_gap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return obj_gap
    
    def get_loss(self, vals, data):
        loss = 0.

        if 'obj' in self.loss_target:
            pred = vals[:, -self.ipm_steps:]
            c_times_x = data.obj_const[:, None] * pred
            obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
            obj_pred = (self.loss_func(obj_pred) * self.step_weight).mean()
            loss = loss + obj_pred

        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']

        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']

        if 'constraint' in self.loss_target:
            constraint_gap = self.get_constraint_violation(vals, data)
            cons_loss = (self.loss_func(constraint_gap) * self.step_weight).mean()
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss

    def get_constraint_violation(self, vals, data):
        """
        Ax - b

        :param vals:
        :param data:
        :return:
        """
        pred = vals[:, -self.ipm_steps:]
        Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
        constraint_gap = Ax - data.rhs[:, None]
        constraint_gap = torch.relu(constraint_gap)
        return constraint_gap

    def get_obj_metric(self, data, pred, hard_non_negative=False):
        # if hard_non_negative, we need a relu to make x all non-negative
        # just for metric usage, not for training
        pred = pred[:, -self.ipm_steps:]
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.obj_const[:, None] * pred
        obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
        x_gt = data.gt_primals[:, -self.ipm_steps:]
        c_times_xgt = data.obj_const[:, None] * x_gt
        obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
        return (obj_pred - obj_gt) / obj_gt

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1.e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "obj_gap"
            }
        }