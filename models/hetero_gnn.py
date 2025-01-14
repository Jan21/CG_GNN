import torch
import torch.nn.functional as F

from models.gcnconv import GCNConv
from models.utils import MLP
from models.hetero_conv import HeteroConv

import math

class ScalarEmbeddingSine1D(torch.nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale
 
  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
 
    pos_x = x_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    return pos_x

def strseq2rank(conv_sequence):
    if conv_sequence == 'parallel':
        c2v = v2c = v2o = o2v = c2o = o2c = 0
    elif conv_sequence == 'cov':
        v2c = o2c = 0
        c2o = v2o = 1
        c2v = o2v = 2
    elif conv_sequence == 'n-o-cl-co-cv':
        co2n = cv2n = 0
        cv2o = 1
        co2cl = 2
        cl2co = n2co = 3
        o2cv = n2cv = 4
    else:
        raise ValueError
    return co2n, cv2n, cv2o, o2cv, co2cl, cl2co, n2co, o2cv, n2cv


def get_conv_layer(conv: str,
                   in_dim: int,
                   hid_dim: int,
                   num_mlp_layers: int,
                   use_norm: bool,
                   in_place: bool):
    if conv.lower() == 'genconv':
        def get_conv():
            return GENConv(in_channels=in_dim,
                           out_channels=hid_dim,
                           num_layers=num_mlp_layers,
                           aggr='softmax',
                           msg_norm=use_norm,
                           learn_msg_scale=use_norm,
                           norm='batch' if use_norm else None,
                           bias=True,
                           edge_dim=1,
                           in_place=in_place)
    elif conv.lower() == 'gcnconv':
        def get_conv():
            return GCNConv(in_dim=in_dim,
                           edge_dim=1,
                           hid_dim=hid_dim,
                           num_mlp_layers=num_mlp_layers,
                           norm='batch' if use_norm else None,
                           in_place=in_place)
    elif conv.lower() == 'ginconv':
        def get_conv():
            return GINEConv(in_dim=in_dim,
                            edge_dim=1,
                            hid_dim=hid_dim,
                            num_mlp_layers=num_mlp_layers,
                            norm='batch' if use_norm else None,
                            in_place=in_place)
    else:
        raise NotImplementedError

    return get_conv


class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
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

        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res
        pe_dim = lappe
        hid_dim = hidden

        if pe_dim > 0:
            self.pe_encoder = torch.nn.ModuleDict({
                'vals': MLP([pe_dim, hid_dim, hid_dim]),
                'cons': MLP([pe_dim, hid_dim, hid_dim]),
                'obj': MLP([pe_dim, hid_dim, hid_dim])})
            in_emb_dim = hid_dim
        else:
            self.pe_encoder = None
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

        if share_lin_weight:
            self.pred_vals = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
            self.pred_cons = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
        else:
            self.pred_vals = torch.nn.ModuleList()
            self.pred_cons = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [2]))
                self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [2]))

        self.node_embed = torch.nn.Linear(hid_dim, in_emb_dim)
        self.pos_embed = ScalarEmbeddingSine1D(hid_dim, normalize=False)
 
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear( hid_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, in_emb_dim ),
        )
    def timestep_embedding(self,timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
    
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    def cat_negative_lits(self, xt, num_vars):
        xt_list = torch.split(xt, num_vars.cpu().numpy().tolist())
        xt_list_with_neg_lits = []
        for x in xt_list:
            xt_list_with_neg_lits.append(torch.cat([x, -x]))
 
        return torch.cat(xt_list_with_neg_lits)    
    
    def forward(self, data,xt,t):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k])
            x_dict[k] = x_emb
        x_l = self.node_embed(self.pos_embed(xt.to(x_dict['vals'].device)))
        x_dict['vals'] = x_l / torch.norm(x_l, dim=1, keepdim=True)
        time_emb = self.time_embed(self.timestep_embedding(t.to(x_l.device), x_l.shape[1]//2))
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
            h['vals'] = h['vals'] + time_emb
            x_dict = h

        cons, vals = zip(*hiddens)

        if self.share_lin_weight:
            vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
            cons = self.pred_cons(torch.stack(cons, dim=0))
            return vals.squeeze().T, cons.squeeze().T
        else:
            vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.num_layers)], dim=1)
            cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.num_layers)], dim=1)
            return vals, cons


class TripartiteHeteroGNNClean(torch.nn.Module):
    def __init__(self,
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

        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res
        pe_dim = lappe
        hid_dim = hidden

        if pe_dim > 0:
            self.pe_encoder = torch.nn.ModuleDict({
                'vals': MLP([pe_dim, hid_dim, hid_dim]),
                'cons': MLP([pe_dim, hid_dim, hid_dim]),
                'obj': MLP([pe_dim, hid_dim, hid_dim])})
            in_emb_dim = hid_dim
        else:
            self.pe_encoder = None
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

        if share_lin_weight:
            self.pred_vals = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
            self.pred_cons = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
        else:
            self.pred_vals = torch.nn.ModuleList()
            self.pred_cons = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
                self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k])
            if self.pe_encoder is not None and hasattr(data[k], 'laplacian_eigenvector_pe'):
                pe_emb = 0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))
                x_emb = torch.cat([x_emb, pe_emb], dim=1)
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

        if self.share_lin_weight:
            vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
            cons = self.pred_cons(torch.stack(cons, dim=0))
            return vals.squeeze().T, cons.squeeze().T
        else:
            vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.num_layers)], dim=1)
            cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.num_layers)], dim=1)
            return vals, cons
        

class ColumnHeteroGNN(torch.nn.Module):
    def __init__(self,
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

        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res
        hid_dim = hidden
        self.pe_encoder = None
        in_emb_dim = 2 * hid_dim

        self.encoder = torch.nn.ModuleDict({'column': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'nodes': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'cvars': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'cliques': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'cobj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch')})


        co2n, cv2n, cv2o, o2cv, co2cl, cl2co, n2co, o2cv, n2cv = strseq2rank(conv_sequence)
        get_conv = get_conv_layer(conv, 2 * hid_dim, hid_dim, num_mlp_layers, use_norm, in_place)
        get_conv_o = get_conv_layer(conv, 2 * hid_dim, 2*hid_dim, num_mlp_layers, use_norm, in_place)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv({
                        ('column','to','nodes'): (get_conv(), co2n),
                        ('nodes', 'to', 'column'): (get_conv(), n2co),
                        ('column', 'to', 'cliques'): (get_conv_o(), co2cl),
                        ('cliques', 'to', 'column'): (get_conv(), cl2co),
                        ('cvars', 'to', 'nodes'): (get_conv(), cv2n),
                        ('nodes', 'to', 'cvars'): (get_conv(), n2cv),
                        ('cvars', 'to', 'cobj'): (get_conv_o(), cv2o),
                        ('cobj', 'to', 'cvars'): (get_conv(), o2cv),
                    }, aggr='cat'))

        self.pred_column = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [2])
        self.pred_cvars = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [2])

        self.node_embed = torch.nn.Linear(hid_dim, in_emb_dim)
        self.pos_embed = ScalarEmbeddingSine1D(hid_dim, normalize=False)
 
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear( hid_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, in_emb_dim ),
        )
    def timestep_embedding(self,timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
    
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    def cat_negative_lits(self, xt, num_vars):
        xt_list = torch.split(xt, num_vars.cpu().numpy().tolist())
        xt_list_with_neg_lits = []
        for x in xt_list:
            xt_list_with_neg_lits.append(torch.cat([x, -x]))
 
        return torch.cat(xt_list_with_neg_lits)    
    
    def forward(self, data,x_column, x_cvars, t_column,t_cvars):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['column', 'nodes', 'cvars', 'cliques', 'cobj']:
            x_emb = self.encoder[k](x_dict[k])
            x_dict[k] = x_emb
        x_cvars = self.node_embed(self.pos_embed(x_cvars.to(x_dict['cvars'].device)))
        x_column = self.node_embed(self.pos_embed(x_column.to(x_dict['column'].device)))
        x_dict['cvars'] = x_cvars / torch.norm(x_cvars, dim=1, keepdim=True)
        x_dict['column'] = x_column / torch.norm(x_column, dim=1, keepdim=True)
        time_emb_cvars = self.time_embed(self.timestep_embedding(t_cvars.to(x_cvars.device), x_cvars.shape[1]//2))
        time_emb_column = self.time_embed(self.timestep_embedding(t_column.to(x_column.device), x_column.shape[1]//2))


        #hiddens = []
        for i in range(self.num_layers):
            if self.share_conv_weight:
                i = 0
            h1 = x_dict
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
            keys = h2.keys()
            #hiddens.append((h2['column'], h2['cvars']))
            if self.use_res:
                h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
            else:
                h = {k: F.relu(h2[k]) for k in keys}
            h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
            h['column'] = h['column'] + time_emb_column
            h['cvars'] = h['cvars'] + time_emb_cvars
            x_dict = h
        pred_column = self.pred_column(h['column'])
        pred_cvars = self.pred_cvars(h['cvars'])
        return pred_column, pred_cvars

