import torch
import pytorch_lightning as pl
import numpy as np
from torch_scatter import scatter
from functools import partial
from models.configs import models_dict
from data.utils import uncollate_fn
import networkx as nx
import random
import time
import multiprocessing
import os # Used for cpu_count if multiprocessing doesn't provide it directly

# --- Core Randomized Greedy Sampling Function ---
def sample_large_is(graph, predictions, alpha=1.0):
    """
    Samples one large independent set using randomized greedy algorithm
    biased by prediction scores.

    Args:
        graph (nx.Graph): The input graph (NetworkX format assumed).
        predictions (dict): A dictionary mapping node -> prediction score (0 to 1).
                            Higher score means more likely to be in a large IS.
        alpha (float): Exponent to adjust weighting (alpha > 1 increases greediness,
                       0 < alpha < 1 increases randomness).

    Returns:
        set: A set representing the vertices in the sampled independent set.
    """

    independent_set = set()
    # Create a copy of the nodes to modify during the process
    available_vertices = set(range(len(predictions)))
    neighbor_dict = graph

    while available_vertices:
        # Get the list of currently available nodes
        current_available_nodes = list(available_vertices)

        # Calculate weights for available vertices based on predictions
        weights = []
        total_weight = 0.0
        for node in current_available_nodes:
            score = predictions[node] # Default to 0 if node not in predictions
            if score < 0 or score > 1:
                 # Optional: Add warning or raise error for invalid scores
                 # print(f"Warning: Node {node} has invalid score {score}. Clamping to [0, 1].")
                 score = max(0.0, min(1.0, score))

            # Use max(score, epsilon) to handle score=0 and alpha!=1 gracefully
            # A very small epsilon prevents issues like 0^0 or division by zero,
            # and gives nodes with 0 score a tiny chance if needed.
            weight = max(score, 1e-9) ** alpha
            weights.append(weight)
            total_weight += weight

        # --- Selection Step ---
        if total_weight <= 1e-9: # If all remaining nodes have effectively zero weight
            # Fallback: Choose uniformly at random from the remaining nodes
            # This can happen if all remaining nodes had score 0 initially
            if not current_available_nodes: # Should not happen if while loop condition is correct
                 break
            chosen_node = random.choice(current_available_nodes)
        else:
            # Choose a node based on the calculated weights
            # random.choices returns a list, so take the first element
            chosen_node = random.choices(current_available_nodes, weights=weights, k=1)[0]

        # --- Update Step ---
        # Add the chosen node to the independent set
        independent_set.add(chosen_node)

        # Identify nodes to remove: the chosen node and all its neighbors
        # Need to handle cases where a neighbor might have already been removed
        neighbors_in_graph = neighbor_dict[chosen_node]
        nodes_to_remove = {chosen_node} | (neighbors_in_graph & available_vertices)

        # Remove them from the set of available vertices for the next iteration
        available_vertices -= nodes_to_remove

    return independent_set

# --- Worker Function for Multiprocessing ---
def run_sampling_task(args):
    """
    Wrapper function to be called by each process in the multiprocessing pool.
    Handles argument unpacking and random seeding for the process.

    Args:
        args (tuple): A tuple containing (graph, predictions, alpha, seed).

    Returns:
        set: The result from sample_large_is.
    """
    graph, predictions, alpha, seed = args
    # IMPORTANT: Seed the random number generator independently in each worker process
    random.seed(seed)
    try:
        result = sample_large_is(graph, predictions, alpha)
        return result
    except Exception as e:
        print(f"Error in worker process (seed {seed}): {e}")
        # Optionally return None or raise the exception depending on desired error handling
        return None



def get_sol_from_preds(val_preds, G):
    # Create a copy of the predictions to avoid modifying the original tensor
    preds_copy = val_preds.clone().detach()
    # Apply softmax to convert logits to probabilities
    # This ensures the prediction scores are properly normalized between 0 and 1
    # which is expected by the sample_large_is function
    preds_copy = torch.nn.functional.softmax(preds_copy, dim=1)
    preds = preds_copy[:,1].cpu().numpy()
    

    # OPTION 2: More efficient multiprocessing
    # Reduce number of samples and workers for large graphs
    num_samples = 40  # Scale samples based on graph size
    alpha_value = 1.5
    
    # Use fewer workers - excessive workers can cause resource contention
    num_workers = min(4, multiprocessing.cpu_count() // 2)
    
    # Create lightweight tasks - minimize data copying
    base_seed = random.randint(0, 2**31)
    
    # Sequential approach for larger graphs that's still efficient
    sampled_sets = []
    largest_set = set()
    
    # Try sequential approach with different seeds
    for i in range(num_samples):
        random.seed(base_seed + i)
        result = sample_large_is(G, preds, alpha_value)
        if len(result) > len(largest_set):
            largest_set = result
    
    # Convert the set of chosen nodes to a binary tensor - use vectorized operation
    binary_solution = torch.zeros(len(G), dtype=torch.long, device=val_preds.device)
    if largest_set:
        # Vectorized assignment is faster than a for loop
        binary_solution[list(largest_set)] = 1
    
    is_size = binary_solution.sum()
    return binary_solution, is_size





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
        self.num_classes = cfg.model.params.num_classes
        self.num_val_iters = cfg.model.params.num_val_iters
        self.num_train_iters = cfg.model.params.num_train_iters
        
    def forward(self, batch, num_iters):
        return self.model(batch, num_iters)


    def compute_loss(self,outputs, batch):
        gt_primals = batch.gt_primals
        outputs = outputs.view(outputs.shape[0], -1, self.num_classes)
        val_preds = outputs[:,-1,:] # Get final predictions
        loss = torch.nn.functional.cross_entropy(val_preds, gt_primals.long())
        return loss
        

    def compute_gap(self,outputs, batch):
        outputs = outputs.view(outputs.shape[0],-1,self.num_classes)
        val_preds = outputs[:,-1,:] #[o[:o.shape[0]//2] for o in outputs["final_truth_assignment"]]
        # Split predictions and labels according to batch indices to handle multiple graphs
        unbatched_preds = uncollate_fn(batch,val_preds)
        sols = []
        is_sizes = []
        for i,preds in enumerate(unbatched_preds):
            graph = batch.graph[i]
            sol, is_size = get_sol_from_preds(preds, graph)
            sols.append(sol)
            is_sizes.append(is_size)
         
        # Create tensor from is_sizes
        is_sizes_tensor = torch.tensor(is_sizes, device=val_preds.device)
        
        # Get the MIS sizes from the batch
        mis_sizes = batch['mis_size']
        
        # Compute the gap between found IS sizes and optimal MIS sizes
        # Gap is the difference between optimal MIS size and our found IS size
        gaps = mis_sizes - is_sizes_tensor
        
        # Compute the average gap
        avg_gap = gaps.float().mean()
        
        # Log the ratio of found IS size to optimal MIS size (as a percentage)
        avg_ratio = (is_sizes_tensor.float() / mis_sizes.float()).mean() * 100.0
        avg_is_size = sum(is_sizes) / len(is_sizes) if is_sizes else 0

        return avg_is_size, avg_gap, avg_ratio


    def training_step(self, data, batch_idx):
        batch_size = data.batch_size
        vals, _ = self(data, self.num_train_iters)
        loss = self.compute_loss(vals, data)  #get_loss(vals, data)
        self.log('train_loss',loss,prog_bar=True, batch_size=batch_size, logger=True)
        return loss
    
    def validation_step(self, data, batch_idx):
        batch_size = data.batch_size
        vals, _ = self(data, self.num_val_iters)
        loss = self.compute_loss(vals, data)
        avg_is_size, avg_gap, avg_ratio = self.compute_gap(vals, data)
        self.log('val_loss',loss, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_avg_is_size',avg_is_size, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_avg_gap',avg_gap, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_avg_ratio',avg_ratio, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True)
        return loss #acc

    def test_step(self, data, batch_idx):
        batch_size = data.batch_size
        vals, _ = self(data)
        acc = self.compute_acc(vals, data)
        self.log('acc', acc, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True, logger=True)
        return acc
    


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
