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
                device,
                ):
        super(Pl_model_wrapper, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        model_class = models_dict[model_name]
        self.model = model_class(**cfg.model.params)
        self.lr = cfg.train.lr
        self.weight_decay = cfg.train.weight_decay
        
    def forward(self, batch):
        return self.model(batch)


    def compute_loss(self,outputs, batch):
        outputs = outputs.view(outputs.shape[0],-1,13)
        val_preds = outputs[:,-1,:] #[o[:o.shape[0]//2] for o in outputs["final_truth_assignment"]]
        node_labels = batch.gt_primals
        loss = torch.nn.functional.cross_entropy(val_preds, node_labels.long().to(val_preds.device))   
        return loss 

    def compute_acc(self, data, vals):
        x0_pred = vals.reshape((1, vals.shape[0], -1, self.model.num_classes))[:,:,-1,:]
        x0_pred_prob = x0_pred.softmax(dim=-1)

        # Calculate accuracy between ground truth and predicted probabilities
        # Convert ground truth to long tensor for comparison
        gt_labels = data.gt_primals.long()
        
        # Get the predicted class (highest probability)
        predicted_classes = torch.argmax(x0_pred_prob, dim=-1)
        
        # Calculate accuracy (correct predictions / total predictions)
        correct_predictions = (predicted_classes == gt_labels).float()
        acc = correct_predictions.mean()
        return acc

    def training_step(self, data, batch_idx):
        batch_size = data.batch_size
        vals, _ = self(data)
        loss = self.compute_loss(vals, data)  #get_loss(vals, data)
        self.log('train_loss',loss,prog_bar=True, batch_size=batch_size, logger=True)
        return loss
    
    def validation_step(self, data, batch_idx):
        batch_size = data.batch_size
        vals, _ = self(data)
        acc = self.compute_acc(vals, data)
        self.log('acc', acc, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True)
        return acc

    def test_step(self, data, batch_idx):
        batch_size = data.batch_size
        vals, _ = self(data)
        acc = self.compute_acc(vals, data)
        self.log('acc', acc, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True)
        return acc
    


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
