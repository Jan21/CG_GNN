import torch
import torch.nn.functional as F

from models.gcnconv import GCNConv
from models.utils import MLP
from models.hetero_conv import HeteroConv
import math

def strseq2rank(conv_sequence):
    if conv_sequence == 'parallel':
        c2v = v2c = v2o = o2v = c2o = o2c = 0
    elif conv_sequence == 'cov':
        v2c = o2c = 0
        c2o = v2o = 1
        c2v = o2v = 2
    else:
        raise ValueError
    return c2v, v2c, v2o, o2v, c2o, o2c


def get_conv_layer(conv: str,
                   in_dim: int,
                   hid_dim: int,
                   num_mlp_layers: int,
                   use_norm: bool,
                   in_place: bool):

    def get_conv():
        return GCNConv(in_dim=in_dim,
                        edge_dim=1,
                        hid_dim=hid_dim,
                        num_mlp_layers=num_mlp_layers,
                        norm='batch' if use_norm else None,
                        in_place=in_place)

    return get_conv


class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 conv,
                 in_shape,
                 lappe,
                 hidden,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 share_conv_weight,
                 share_lin_weight,
                 use_norm,
                 use_res,
                 num_iters,
                 in_place=True,
                 conv_sequence='parallel'):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res
        hid_dim = hidden
        in_emb_dim = 2 * hid_dim

        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch')})

        c2v, v2c, v2o, o2v, c2o, o2c = strseq2rank(conv_sequence)
        get_conv = get_conv_layer(conv, 2 * hid_dim, hid_dim, num_mlp_layers, use_norm, in_place)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): (get_conv(), c2v),
                        ('vals', 'to', 'cons'): (get_conv(), v2c),
                        ('vals', 'to', 'obj'): (get_conv(), v2o),
                        ('obj', 'to', 'vals'): (get_conv(), o2v),
                        ('cons', 'to', 'obj'): (get_conv(), c2o),
                        ('obj', 'to', 'cons'): (get_conv(), o2c),
                    }, aggr='cat'))


        self.pred_vals = torch.nn.ModuleList()
        self.pred_cons = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [self.num_classes]))
            self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [self.num_classes]))
 

    
    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k])
            x_dict[k] = x_emb

        hiddens = []
        for i in range(self.num_layers):
            if self.share_conv_weight:
                i = 0
            h1 = x_dict
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
            keys = h2.keys()
            hiddens.append((h2['cons'], h2['vals']))
            if self.use_res:
                h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
            else:
                h = {k: F.relu(h2[k]) for k in keys}
            h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
            x_dict = h

        cons, vals = zip(*hiddens)
        vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.num_layers)], dim=1)
        cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.num_layers)], dim=1)
        return vals, cons
        




