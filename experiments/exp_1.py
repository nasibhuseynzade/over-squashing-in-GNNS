# First experiment
# Calculating commute time changes of original dataset and modified dataset


from .commute_time import aggregate_commute_times
from torch_geometric.datasets import ZINC, QM9
from torch_geometric.utils import to_networkx
from methods._fosr import apply_fosr
import numpy as np

import networkx as nx
from math import inf
from numba import jit, int64



def process_dataset(dataset_name, num_graphs=1000):
    if dataset_name.lower() == 'zinc':
        dataset = ZINC(root='/tmp/ZINC', subset=False)
    elif dataset_name.lower() == 'qm9':
        dataset = QM9(root='/tmp/QM9')
    else:
        raise ValueError("Invalid dataset name. Choose 'ZINC' or 'QM9'.")
    
    results = []
    
    for i in range(min(num_graphs, len(dataset))):
        data = dataset[i]
        
        # Convert to NetworkX graph
        G_original = to_networkx(data, to_undirected=True)
        
        # Get edge index from the graph
        edge_index = np.array(list(G_original.edges())).T
        
        # Apply FOSR to get new edge index
        new_edge_index, new_edge_type = apply_fosr(edge_index)
        
        # Create a new graph with the rewired edges
        G_fosr = nx.Graph()
        G_fosr.add_nodes_from(range(data.num_nodes))
        new_edges = list(zip(new_edge_index[0], new_edge_index[1]))
        G_fosr.add_edges_from(new_edges)
        
        # Calculate aggregate commute times for both graphs
        agg_commute_time_original = aggregate_commute_times(G_original)
        agg_commute_time_fosr = aggregate_commute_times(G_fosr)
        
        # Calculate the percentage change in aggregate commute time
        percent_change = ((agg_commute_time_fosr - agg_commute_time_original) / agg_commute_time_original) * 100
        
        results.append({
            'graph_index': i,
            'original_edges': G_original.number_of_edges(),
            'fosr_edges': G_fosr.number_of_edges(),
            'added_edges': G_fosr.number_of_edges() - G_original.number_of_edges(),
            'original_commute_time': agg_commute_time_original,
            'fosr_commute_time': agg_commute_time_fosr,
            'percent_change': percent_change
        })
        
        if i % 1000 == 0:
            print(f"Processed {i+1} graphs...")
    
    return results

# Process ZINC dataset
print("Processing ZINC dataset...")
zinc_results = process_dataset('ZINC')

# Process QM9 dataset
print("\nProcessing QM9 dataset...")
qm9_results = process_dataset('QM9')

# Function to print summary statistics
def print_summary(results, dataset_name):
    avg_original_edges = np.mean([r['original_edges'] for r in results])
    avg_fosr_edges = np.mean([r['fosr_edges'] for r in results])
    avg_added_edges = np.mean([r['added_edges'] for r in results])
    avg_original_commute_time = np.mean([r['original_commute_time'] for r in results])
    avg_fosr_commute_time = np.mean([r['fosr_commute_time'] for r in results])
    avg_percent_change = np.mean([r['percent_change'] for r in results])
    
    print(f"\nSummary for {dataset_name} dataset:")
    print(f"Average original edges: {avg_original_edges:.2f}")
    print(f"Average FOSR edges: {avg_fosr_edges:.2f}")
    print(f"Average added edges: {avg_added_edges:.2f}")
    print(f"Average original commute time: {avg_original_commute_time:.4f}")
    print(f"Average FOSR commute time: {avg_fosr_commute_time:.4f}")
    print(f"Average percentage change in commute time: {avg_percent_change:.2f}%")

# Print summaries
print_summary(zinc_results, "ZINC")
print_summary(qm9_results, "QM9")