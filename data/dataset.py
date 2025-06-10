import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset, Dataset
from torch_sparse import SparseTensor
from tqdm import tqdm
import networkx as nx

class ProblemsDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        extra_path: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.using_ineq = True
        self.extra_path = extra_path
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['instance_0.pkl.gz']   # there should be at least one pkg

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_' + self.extra_path)

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def prepare_example(self,A,b,c,x):
        sp_a = SparseTensor.from_dense(A, has_value=True)

        row = sp_a.storage._row
        col = sp_a.storage._col
        val = sp_a.storage._value

        if self.using_ineq:
            tilde_mask = torch.ones(row.shape, dtype=torch.bool)
        else:
            tilde_mask = col < (A.shape[1] - A.shape[0])

        c = c / (c.abs().max() + 1.e-10)  # does not change the result


        gt_primals = x.to(torch.long) #torch.from_numpy(x).to(torch.float)
            # gt_duals = torch.from_numpy(l).to(torch.float)
            # gt_slacks = torch.from_numpy(s).to(torch.float)

        data = HeteroData(
            cons={'x': torch.cat([A.mean(1, keepdims=True),
                                    A.std(1, keepdims=True)], dim=1)},
            vals={'x': torch.cat([A.mean(0, keepdims=True),
                                    A.std(0, keepdims=True)], dim=0).T},
            obj={'x': torch.cat([c.mean(0, keepdims=True),
                                    c.std(0, keepdims=True)], dim=0)[None]},

            cons__to__vals={'edge_index': torch.vstack(torch.where(A)),
                            'edge_attr': A[torch.where(A)][:, None]},
            vals__to__cons={'edge_index': torch.vstack(torch.where(A.T)),
                            'edge_attr': A.T[torch.where(A.T)][:, None]},
            vals__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[1]),
                                                        torch.zeros(A.shape[1], dtype=torch.long)]),
                            'edge_attr': c[:, None]},
            obj__to__vals={'edge_index': torch.vstack([torch.zeros(A.shape[1], dtype=torch.long),
                                                        torch.arange(A.shape[1])]),
                            'edge_attr': c[:, None]},
            cons__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[0]),
                                                        torch.zeros(A.shape[0], dtype=torch.long)]),
                            'edge_attr': b[:, None]},
            obj__to__cons={'edge_index': torch.vstack([torch.zeros(A.shape[0], dtype=torch.long),
                                                        torch.arange(A.shape[0])]),
                            'edge_attr': b[:, None]},
            gt_primals=gt_primals,
            # gt_duals=gt_duals,
            # gt_slacks=gt_slacks,
            obj_value=0, #torch.tensor(sol['fun'].astype(np.float32)),
            obj_const=c,

            A_row=row,
            A_col=col,
            A_val=val,
            A_num_row=A.shape[0],
            A_num_col=A.shape[1],
            A_nnz=len(val),
            A_tilde_mask=tilde_mask,
            rhs=b)

        if self.pre_filter is not None:
            raise NotImplementedError

        if self.pre_transform is not None:
            data = data #self.pre_transform(data)
        return data

    def process(self):
        num_instance_pkg = len([n for n in os.listdir(self.raw_dir) if n.endswith('pkl.gz')])

        data_list = []
        for i in range(num_instance_pkg):
            # load instance
            print(f"processing {i}th package, {num_instance_pkg} in total")
            with gzip.open(os.path.join(self.raw_dir, f"instance_{i}.pkl.gz"), "rb") as file:
                ip_pkgs = pickle.load(file)

            for ip_idx in tqdm(range(len(ip_pkgs))):
                (A, b, c, x) = ip_pkgs[ip_idx]
                data = self.prepare_example(A, b, c, x)
                data_list.append(data)

            torch.save(Batch.from_data_list(data_list), osp.join(self.processed_dir, f'batch{i}.pt'))
            data_list = []

        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))



class LargeProblemDataset(Dataset):
    def __init__(
        self,
        root: str,
        extra_path: str,
        upper_bound: Optional = None,
        rand_starts: int = 1,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.using_ineq = True
        self.extra_path = extra_path
        self.upper_bound = upper_bound
        self.rand_starts = rand_starts
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Create an index of all saved examples
        self.data_files = []
        processed_dir = self.processed_dir
        for file_name in os.listdir(processed_dir):
            if file_name.endswith('.pt') and file_name.startswith('data_'):
                self.data_files.append(file_name)
        
        self.data_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
    @property
    def raw_file_names(self) -> List[str]:
        return ['instance_0.pkl.gz']  # there should be at least one pkg

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_' + self.extra_path)

    @property
    def processed_file_names(self) -> List[str]:
        # Return a placeholder to make PyTorch Geometric happy
        # We'll handle the file management ourselves
        return ['metadata.pt']

    def prepare_example(self,A,b,c,sol, graph, mis_size):
                #sp_a = A #SparseTensor.from_dense(A, has_value=True)
                # Convert A from sparse pytorch tensor to SparseTensor from torch_sparse
                sp_a = SparseTensor.from_dense(A, has_value=True)

                row = sp_a.storage._row
                col = sp_a.storage._col
                val = sp_a.storage._value

                if self.using_ineq:
                    tilde_mask = torch.ones(row.shape, dtype=torch.bool)
                else:
                    tilde_mask = col < (A.shape[1] - A.shape[0])

                c = c / (c.abs().max() + 1.e-10)  # does not change the result

                # solve the LP
                # if self.using_ineq:
                #     A_ub = A.numpy()
                #     b_ub = b.numpy()
                #     A_eq = None
                #     b_eq = None
                # else:
                #     A_eq = A.numpy()
                #     b_eq = b.numpy()
                #     A_ub = None
                #     b_ub = None
                bounds = (0, self.upper_bound)

                for _ in range(self.rand_starts):
                    gt_primals = sol
                    data = HeteroData(
                        cons={'x': torch.cat([A.mean(1, keepdims=True),
                                              A.std(1, keepdims=True)], dim=1)},
                        vals={'x': torch.cat([A.mean(0, keepdims=True),
                                              A.std(0, keepdims=True)], dim=0).T},
                        obj={'x': torch.cat([c.mean(0, keepdims=True),
                                             c.std(0, keepdims=True)], dim=0)[None]},
                        cons__to__vals={'edge_index': torch.vstack(torch.where(A)),
                                        'edge_attr': A[torch.where(A)][:, None]},
                        vals__to__cons={'edge_index': torch.vstack(torch.where(A.T)),
                                        'edge_attr': A.T[torch.where(A.T)][:, None]},
                        vals__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[1]),
                                                                   torch.zeros(A.shape[1], dtype=torch.long)]),
                                       'edge_attr': c[:, None]},
                        obj__to__vals={'edge_index': torch.vstack([torch.zeros(A.shape[1], dtype=torch.long),
                                                                   torch.arange(A.shape[1])]),
                                       'edge_attr': c[:, None]},
                        cons__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[0]),
                                                                   torch.zeros(A.shape[0], dtype=torch.long)]),
                                       'edge_attr': b[:, None]},
                        obj__to__cons={'edge_index': torch.vstack([torch.zeros(A.shape[0], dtype=torch.long),
                                                                   torch.arange(A.shape[0])]),
                                       'edge_attr': b[:, None]},
                        gt_primals=gt_primals,
                        # gt_duals=gt_duals,
                        # gt_slacks=gt_slacks,
                        obj_value=0, #torch.tensor(sol['fun'].astype(np.float32)),
                        obj_const=c,

                        A_row=row,
                        A_col=col,
                        A_val=val,
                        A_num_row=A.shape[0],
                        A_num_col=A.shape[1],
                        A_nnz=len(val),
                        A_tilde_mask=tilde_mask,
                        rhs=b,
                        graph=graph,
                        mis_size=mis_size)

                    if self.pre_filter is not None:
                        raise NotImplementedError

                    if self.pre_transform is not None:
                        data = data #self.pre_transform(data)
                    return data

    def get_graph(self, A):
        G = nx.Graph()
        G.add_nodes_from(range(A.shape[1]))

        # For each pair of vertices, check if they share any cliques
        # If they do, add an edge between them
        for i in range(A.shape[1]):
            for j in range(i+1, A.shape[1]):
                # Get the cliques containing vertex i and j
                cliques_i = set(np.where(A[:,i] == 1)[0])
                cliques_j = set(np.where(A[:,j] == 1)[0])
                
                # If vertices share any cliques, they are connected
                if len(cliques_i.intersection(cliques_j)) > 0:
                    G.add_edge(i,j)
        # Create a dictionary mapping each node to its neighbors
        neighbor_dict = {node: set(G.neighbors(node)) for node in G.nodes()}
        return neighbor_dict


    def process(self):
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Dictionary to store metadata
        metadata = {'num_examples': 0, 'example_paths': []}
        
        example_idx = 0
        num_instance_pkg = len([n for n in os.listdir(self.raw_dir) if n.endswith('pkl.gz')])

        for i in range(num_instance_pkg):
            print(f"processing {i}th package, {num_instance_pkg} in total")
            # Load instance
            with gzip.open(os.path.join(self.raw_dir, f"instance_{i}.pkl.gz"), "rb") as file:
                ip_pkgs = pickle.load(file)

            for ip_idx in tqdm(range(len(ip_pkgs))):
                (A, b, c, sol) = ip_pkgs[ip_idx]
                graph = self.get_graph(A)
                mis_size = sol.sum()
                data = self.prepare_example(A, b, c, sol, graph, mis_size)
                
                # Apply pre-filter
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                
                # Apply pre-transform
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                # Save individual example
                example_path = os.path.join(self.processed_dir, f'data_{example_idx}.pt')
                torch.save(data, example_path)
                
                # Update metadata
                metadata['num_examples'] += 1
                metadata['example_paths'].append(example_path)
                example_idx += 1

        # Save metadata
        torch.save(metadata, os.path.join(self.processed_dir, 'metadata.pt'))

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        # Load a single data point from disk
        data_path = os.path.join(self.processed_dir, self.data_files[idx])
        data = torch.load(data_path)
        
        #if self.transform:
        #    data = self.transform(data)
            
        return data