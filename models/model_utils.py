import torch
import pytorch_lightning as pl
import numpy as np
from torch_scatter import scatter
from functools import partial
from models.configs import models_dict
from models.diffusion_utils import CategoricalDiffusion, InferenceSchedule, prepare_diffusion


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
        self.diffusion = CategoricalDiffusion(T=1000, schedule='linear')
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


    def compute_diffusion_loss(self,outputs, batch):
        positive_lit_otp = [o[:o.shape[0]//2] for o in outputs["final_truth_assignment"]]
        products = torch.cat(positive_lit_otp, dim = 0)
        node_labels =((batch.assignment + 1)/2)
        loss = torch.nn.functional.cross_entropy(products, node_labels.long().to(products.device))   
        return loss 

    def training_step(self, data, batch_idx):
        t, xt = prepare_diffusion(data, self.diffusion)
        vals, _ = self.model(data, self.num_iters, xt, t)
        #vals, _ = self(data)
        loss = self.compute_diffusion_loss(vals, data)
        batch_size = data.batch_size
        #loss = self.get_loss(vals, data)
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
        Ax = scatter(pred[data.A_col, -1] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
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
    

    def categorical_denoise_step(self, xt, t, device, batch, target_t=None):
      with torch.no_grad():
        t = torch.from_numpy(t).view(1)
        x0_pred = self.model(
            batch,
            self.num_iters,
            xt.float(),
            t.float(),
        )
        xt = xt.to(device)
        t = t.to(device)
        x0_pred = x0_pred["final_truth_assignment"][0]
        x0_pred = x0_pred[:x0_pred.shape[0]//2, :]
        x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
        xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
        return xt
    
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
      
      """Sample from the categorical posterior for a given time step.
        See https://arxiv.org/pdf/2107.03006.pdf for details.
      """
      diffusion = self.diffusion
 
      if target_t is None:
        target_t = t - 1
      else:
        target_t = torch.from_numpy(target_t).view(1)
 
      # Thanks to Daniyar and Shengyu, who found the "target_t == 0" branch is not needed :)
      # if target_t > 0:
      Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
      Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
      # else:
      #   Q_t = torch.eye(2).float().to(x0_pred_prob.device)
      Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
      Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)
 
      xt = torch.nn.functional.one_hot(xt.long(), num_classes=2).float()
      xt = xt.reshape(x0_pred_prob.shape)
 
      x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
      x_t_target_prob_part_2 = Q_bar_t_target[0]
      x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)
 
      x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3
 
      sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
      x_t_target_prob_part_2_new = Q_bar_t_target[1]
      x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)
 
      x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new
 
      sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]
 
      if target_t > 0:
        xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
      else:
        xt = sum_x_t_target_prob.clamp(min=0)
 
      """
      if self.sparse:
        xt = xt.reshape(-1)
      """
      return xt
   
    def validation_diffusion(self, batch):
        stacked_predict_labels = []
        device = batch.x_l.device
        batch_size = 1
        steps = 50
        num_solutions = 1
      
        for _ in range(num_solutions):
          node_labels = (batch.assignment.cpu() + 1)/2
          xt = torch.randn_like(node_labels.float())
          xt = (xt > 0).long()
          xt = xt.reshape(-1)
 
          time_schedule = InferenceSchedule(inference_schedule="cosine",
                                          T=self.diffusion.T, inference_T=steps)
          for i in range(steps):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
            t2 = np.array([t2 for _ in range(batch_size)]).astype(int)
 
            xt = self.categorical_denoise_step(xt, t1, device, batch, target_t=t2)
            xt = xt.squeeze(2).squeeze(0)
 
          predict_labels = xt.float().cpu().detach().numpy() + 1e-6
          stacked_predict_labels.append(predict_labels)        
 
          infered_assignment = np.round(stacked_predict_labels[-1])
          assert not np.any((infered_assignment !=0)&(infered_assignment !=1))
          infered_assignment = infered_assignment * 2 - 1
 
          result_literals = []
          for ix, assignment in enumerate(list(infered_assignment)):
              result_literals.append(int((ix+1) * assignment))
          
          sat_num = 0
          for c in batch.clauses[0]:
              for lit in c:
                  if lit in result_literals:
                      sat_num +=1
                      break
          
          gap = len(batch.clauses[0])-sat_num
          if gap == 0:
             acc = 1
          else:
             acc = 0
 
          
          self.log('val_avg_gap', gap, prog_bar=True, logger=True)
          self.log('val_accuracy', acc, prog_bar=True, logger=True)
 
        return 1