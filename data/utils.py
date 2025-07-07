from typing import Dict, List
import math

import torch
import numpy as np
from torch_geometric.data import Data, Batch


def log_normalize(x):
    return torch.log(1. + x)


def log_denormalize(x):
    return torch.exp(x) - 1.


def mode_of_distribution(x):
    cnt, intervals = np.histogram(x, bins=50, range=None, density=None, weights=None)
    idx = cnt.argmax()
    return (intervals[idx] + intervals[idx + 1]) / 2


def args_set_bool(args: Dict):
    for k, v in args.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                args[k] = True
            elif v.lower() == 'false':
                args[k] = False
    return args


def barrier_function(x, t=1.e5):
    cond = x.detach() >= 1 / (t ** 2)
    return torch.where(cond, (-1 / t) * torch.log(x), -t * x - 1 / t * math.log(1 / (t ** 2)) + 1 / t)




def collate_fn_ip(graphs: List[Data]):
    # Add graph_idx to each node before batching
    edges = []
    for i, graph in enumerate(graphs):
        num_nodes = graph.A_num_col  # Number of variables/columns
        graph.node_graph_idx = torch.full((num_nodes,), i, dtype=torch.long)
        edges.append(graph['graph'])
        del graph['graph']
    
    new_batch = Batch.from_data_list(graphs)
    
    # Process rows (constraints)
    row_bias = torch.hstack([new_batch.A_num_row.new_zeros(1), new_batch.A_num_row[:-1]]).cumsum(dim=0)
    row_bias = torch.repeat_interleave(row_bias, new_batch.A_nnz)
    new_batch.A_row += row_bias
    
    # Process columns (variables)
    col_bias = torch.hstack([new_batch.A_num_col.new_zeros(1), new_batch.A_num_col[:-1]]).cumsum(dim=0)
    col_bias = torch.repeat_interleave(col_bias, new_batch.A_nnz)
    new_batch.A_col += col_bias
    
    new_batch.graph = edges
    return new_batch


def uncollate_fn(batch,preds):
    graph_ids = batch.node_graph_idx
    costs = batch.obj_const
    primals = batch.gt_primals
    # Split predictions according to graph IDs
    unique_graph_ids = torch.unique(graph_ids)
    split_preds = []
    split_costs = []
    objs = []
    
    for graph_id in unique_graph_ids:
        # Create mask for current graph ID
        mask = (graph_ids == graph_id)
        # Extract predictions for this graph
        graph_preds = preds[mask]
        graph_costs = costs[mask]
        graph_primals = primals[mask]
        split_preds.append(graph_preds)
        split_costs.append(graph_costs)
        # Compute inner product between primals and costs for this graph
        obj = (graph_primals * graph_costs).sum()
        objs.append(obj)
    
    return split_preds, split_costs, objs

