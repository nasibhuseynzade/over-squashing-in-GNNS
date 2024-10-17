# First experiment
# Calculating commute time changes of original dataset and modified dataset


from metrics.commute_time import aggregate_commute_times
from methods._fosr import apply_fosr
from torch_geometric.datasets import ZINC, QM9
from torch_geometric.utils import to_networkx
from methods import _fosr
import numpy as np
import networkx as nx
from math import inf
from numba import jit, int64

def process_dataset(dataset_name):
    if dataset_name.lower() == 'zinc':
        dataset = ZINC(root='/tmp/ZINC', subset=False)  # Using the full dataset
    elif dataset_name.lower() == 'qm9':
        dataset = QM9(root='/tmp/QM9')
    else:
        raise ValueError("Invalid dataset name. Choose 'ZINC' or 'QM9'.")
    
    total_original_edges = 0
    total_fosr_edges = 0
    total_graphs = len(dataset)
    
    for i, data in enumerate(dataset):
        # Convert to NetworkX graph
        G_original = to_networkx(data, to_undirected=True)
        
        # Get edge index from the graph
        edge_index = np.array(list(G_original.edges())).T
        
        # Apply FOSR to get new edge index
        new_edge_index, _ = apply_fosr(edge_index)
        
        # Create a new graph with the rewired edges
        G_fosr = nx.Graph()
        G_fosr.add_nodes_from(range(data.num_nodes))
        new_edges = list(zip(new_edge_index[0], new_edge_index[1]))
        G_fosr.add_edges_from(new_edges)
        
        total_original_edges += G_original.number_of_edges()
        total_fosr_edges += G_fosr.number_of_edges()
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{total_graphs} graphs...")
    
    return total_original_edges, total_fosr_edges, total_graphs

# Function to print summary statistics
def print_summary(original_edges, fosr_edges, total_graphs, dataset_name):
    print(f"\nSummary for {dataset_name} dataset:")
    print(f"Total graphs processed: {total_graphs}")
    print(f"Number of original edges: {original_edges}")
    print(f"Number of edges in FOSR dataset: {fosr_edges}")
    print(f"Difference in edges: {fosr_edges - original_edges}")

# Process ZINC dataset
print("Processing ZINC dataset...")
zinc_original, zinc_fosr, zinc_graphs = process_dataset('ZINC')

# Process QM9 dataset
print("\nProcessing QM9 dataset...")
qm9_original, qm9_fosr, qm9_graphs = process_dataset('QM9')

# Print summaries
print_summary(zinc_original, zinc_fosr, zinc_graphs, "ZINC")
print_summary(qm9_original, qm9_fosr, qm9_graphs, "QM9")