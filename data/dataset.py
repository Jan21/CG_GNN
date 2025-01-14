import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset
from torch_sparse import SparseTensor
from solver.ilp import solve_ilp
from solver.linprog import linprog
from tqdm import tqdm


class LPDataset(InMemoryDataset):

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
        self.rand_starts = rand_starts
        self.using_ineq = True
        self.extra_path = extra_path
        self.upper_bound = upper_bound
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

    def prepare_example(self,A,b,c):
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
                if self.using_ineq:
                    A_ub = A.numpy()
                    b_ub = b.numpy()
                    A_eq = None
                    b_eq = None
                else:
                    A_eq = A.numpy()
                    b_eq = b.numpy()
                    A_ub = None
                    b_ub = None

                bounds = (0, self.upper_bound)

                for _ in range(self.rand_starts):
                    # sol = ipm_overleaf(c.numpy(), A_ub, b_ub, A_eq, b_eq, None, max_iter=1000, lin_solver='scipy_cg')
                    # x = np.stack(sol['xs'], axis=1)  # primal

                    sol = linprog(c.numpy(),
                                  A_ub=A_ub,
                                  b_ub=b_ub,
                                  A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                                  method='interior-point', callback=lambda res: res.x)
                    if sol == None:
                        continue
                    x = np.stack(sol.intermediate, axis=1)
                    assert not np.isnan(sol['fun'])

                    gt_primals = torch.from_numpy(x).to(torch.float)
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
                        obj_value=torch.tensor(sol['fun'].astype(np.float32)),
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
                        data = self.pre_transform(data)
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
                (A, b, c) = ip_pkgs[ip_idx]
                data = self.prepare_example(A, b, c)
                data_list.append(data)

            torch.save(Batch.from_data_list(data_list), osp.join(self.processed_dir, f'batch{i}.pt'))
            data_list = []

        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))



class ILPDataset(InMemoryDataset):

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
        self.rand_starts = rand_starts
        self.using_ineq = True
        self.extra_path = extra_path
        self.upper_bound = upper_bound
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

    def prepare_example(self,ex):
        column__to__cliques_matrix = ex['graph_matrix'].T
        cliques = ex['b']
        cvars__to__nodes_matrix = ex['available'].T
        cvars_labels = ex['available_labels']
        column_labels = ex['target']
        num_vars = len(column_labels)
        column__to__nodes_matrix = torch.eye(num_vars)
        c = torch.ones(len(cvars_labels))

        sp_column__to__cliques_matrix = SparseTensor.from_dense(column__to__cliques_matrix, has_value=True)
        sp_cvars__to__nodes_matrix = SparseTensor.from_dense(cvars__to__nodes_matrix, has_value=True)
        sp_column__to__nodes_matrix = SparseTensor.from_dense(column__to__nodes_matrix, has_value=True)

        column__to__cliques_row = sp_column__to__cliques_matrix.storage._row
        column__to__cliques_col = sp_column__to__cliques_matrix.storage._col
        column__to__cliques_val = sp_column__to__cliques_matrix.storage._value

        cvars__to__nodes_row = sp_cvars__to__nodes_matrix.storage._row
        cvars__to__nodes_col = sp_cvars__to__nodes_matrix.storage._col
        cvars__to__nodes_val = sp_cvars__to__nodes_matrix.storage._value

        column__to__nodes_row = sp_column__to__nodes_matrix.storage._row
        column__to__nodes_col = sp_column__to__nodes_matrix.storage._col
        column__to__nodes_val = sp_column__to__nodes_matrix.storage._value
        # Create a matrix that sums each column of A when multiplied with flattened A
        #graph_matrix_num_rows, graph_matrix_num_cols = graph_matrix.shape
        #sum_column_matrix = torch.zeros((num_cols, num_rows * num_cols))
        #for i in range(num_cols):
        #    sum_column_matrix[i, i*num_rows:(i+1)*num_rows] = 1
        # Create a matrix that sums each row of A when multiplied with flattened A
        #sum_row_matrix = torch.zeros((num_rows, num_rows * num_cols))
        #for i in range(num_rows):
        #    sum_row_matrix[i, i::num_rows] = 1


        if self.using_ineq:
            cvars__to__nodes_tilde_mask = torch.ones(cvars__to__nodes_row.shape, dtype=torch.bool)
            column__to__cliques_tilde_mask = torch.ones(column__to__cliques_row.shape, dtype=torch.bool)
            column__to__nodes_tilde_mask = torch.ones(column__to__nodes_row.shape, dtype=torch.bool)
        else:
            columns_matrix_tilde_mask = columns_matrix_col < (graph_matrix_num_cols - graph_matrix_num_rows)



        #sol,obj_val = solve_ilp(c=c.numpy(), A=A_eq, b=b_eq)

        #cons1_vals = torch.from_numpy(sol).to(torch.float)
        # gt_duals = torch.from_numpy(l).to(torch.float)
        # gt_slacks = torch.from_numpy(s).to(torch.float)

        data = HeteroData(
            cvars={'x': torch.cat([cvars__to__nodes_matrix.mean(1, keepdims=True),
                                    cvars__to__nodes_matrix.std(1, keepdims=True)], dim=1)},
            nodes={'x': torch.cat([cvars__to__nodes_matrix.mean(0, keepdims=True),
                                    cvars__to__nodes_matrix.std(0, keepdims=True)], dim=0).T},
            column={'x': torch.cat([column__to__cliques_matrix.mean(1, keepdims=True),
                                    torch.zeros_like(column__to__cliques_matrix.std(1, keepdims=True))],
                                    #A.std(0, keepdims=True)],
                                    dim=1)},
            cliques = {'x': torch.cat([column__to__cliques_matrix.mean(0, keepdims=True),
                                    column__to__cliques_matrix.std(0, keepdims=True)], dim=0).T},
            cobj={'x': torch.cat([c.mean(0, keepdims=True),
                                    c.std(0, keepdims=True)], dim=0).T[None]},


            column__to__nodes={'edge_index': torch.vstack(torch.where(column__to__nodes_matrix)),
                            'edge_attr': column__to__nodes_matrix[torch.where(column__to__nodes_matrix)][:, None]},
            nodes__to__column={'edge_index': torch.vstack(torch.where(column__to__nodes_matrix.T)),
                            'edge_attr': column__to__nodes_matrix.T[torch.where(column__to__nodes_matrix.T)][:, None]},
            column__to__cliques={'edge_index': torch.vstack(torch.where(column__to__cliques_matrix)),
                            'edge_attr': column__to__cliques_matrix[torch.where(column__to__cliques_matrix)][:, None]},
            cliques__to__column={'edge_index': torch.vstack(torch.where(column__to__cliques_matrix.T)),
                            'edge_attr': column__to__cliques_matrix.T[torch.where(column__to__cliques_matrix.T)][:, None]},
            cvars__to__nodes={'edge_index': torch.vstack(torch.where(cvars__to__nodes_matrix)),
                            'edge_attr': cvars__to__nodes_matrix[torch.where(cvars__to__nodes_matrix)][:, None]},
            nodes__to__cvars={'edge_index': torch.vstack(torch.where(cvars__to__nodes_matrix.T)),
                            'edge_attr': cvars__to__nodes_matrix.T[torch.where(cvars__to__nodes_matrix.T)][:, None]},
            cvars__to__cobj={'edge_index': torch.vstack([torch.arange(len(cvars_labels)),
                                                        torch.zeros(len(cvars_labels), dtype=torch.long)]),
                            'edge_attr': c[:, None]},
            cobj__to__cvars={'edge_index': torch.vstack([torch.zeros(len(cvars_labels), dtype=torch.long),
                                                        torch.arange(len(cvars_labels))]),
                            'edge_attr': c[:, None]},
            cvars_labels=cvars_labels,
            num_cvars=len(cvars_labels),
            column_labels=column_labels,
            num_column=len(column_labels),

            column__to__cliques_row=column__to__cliques_row,
            column__to__cliques_col=column__to__cliques_col,
            column__to__cliques_val=torch.tensor(column__to__cliques_val),
            column__to__cliques_num_row=column__to__cliques_matrix.shape[0],
            column__to__cliques_num_col=column__to__cliques_matrix.shape[1],
            column__to__cliques_nnz=len(column__to__cliques_val),

            cvars__to__nodes_row=cvars__to__nodes_row,
            cvars__to__nodes_col=cvars__to__nodes_col,
            cvars__to__nodes_val=torch.tensor(cvars__to__nodes_val),
            cvars__to__nodes_num_row=cvars__to__nodes_matrix.shape[0],
            cvars__to__nodes_num_col=cvars__to__nodes_matrix.shape[1],
            cvars__to__nodes_nnz=len(cvars__to__nodes_val),

            column__to__nodes_row=column__to__nodes_row,
            column__to__nodes_col=column__to__nodes_col,
            column__to__nodes_val=torch.tensor(column__to__nodes_val),
            column__to__nodes_num_row=column__to__nodes_matrix.shape[0],
            column__to__nodes_num_col=column__to__nodes_matrix.shape[1],
            column__to__nodes_nnz=len(column__to__nodes_val)
            )

        if self.pre_filter is not None:
                raise NotImplementedError

        if self.pre_transform is not None:
                raise NotImplementedError
        return data
    
    def process(self):
        num_instance_pkg = len([n for n in os.listdir(self.raw_dir) if n.endswith('pkl.gz')])

        data_list = []
        for i in range(num_instance_pkg):
            # load instance
            batch_path = osp.join(self.processed_dir, f'batch{i}.pt')
            if osp.exists(batch_path):
                print(f"batch {i} already exists")
                continue
            print(f"processing {i}th package, {num_instance_pkg} in total")
            with gzip.open(os.path.join(self.raw_dir, f"instance_{i}.pkl.gz"), "rb") as file:
                ip_pkgs = pickle.load(file)

            for ip_idx in tqdm(range(len(ip_pkgs))):
                ex = ip_pkgs[ip_idx]
                data = self.prepare_example(ex)
                data_list.append(data)

            torch.save(Batch.from_data_list(data_list), osp.join(self.processed_dir, f'batch{i}.pt'))
            data_list = []

        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))
