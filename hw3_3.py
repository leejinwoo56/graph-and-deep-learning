import numpy as np
import sys

# Graph class
class Graph:
    def __init__(self):
        self.adj_list = {}
        self.nodes = set()
    
    def add_edge(self, u, v):
        if u not in self.adj_list: self.adj_list[u] = []
        if v not in self.adj_list: self.adj_list[v] = []
        
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)
        self.nodes.add(u)
        self.nodes.add(v)
        
    def neighbors(self, node):
        if node in self.adj_list:
            return sorted(self.adj_list[node])
        return []
    
    def get_max_node_id(self):
        if not self.nodes: return 0
        return max(self.nodes)

def node2vec_walk(graph, start, length=5, p=1.0, q=1.0):
    walk = [start]
    if length <= 1: return walk
        
    curr = start
    prev = None
    
    # 1st Step
    neighbors = graph.neighbors(curr)
    if not neighbors: return walk
        
    prev = curr
    curr = neighbors[0] # Smallest index
    walk.append(curr)
    
    # Subsequent Steps
    for _ in range(length - 2):
        neighbors = graph.neighbors(curr)
        if not neighbors: break
            
        prev_neighbors = set(graph.neighbors(prev))
        best_node = -1
        max_prob = -1.0
        
        for neighbor in neighbors:
            weight = 0.0
            if neighbor == prev: weight = 1.0 / p
            elif neighbor in prev_neighbors: weight = 1.0
            else: weight = 1.0 / q
            
            if weight > max_prob:
                max_prob = weight
                best_node = neighbor
        
        prev = curr
        curr = best_node
        walk.append(curr)
        
    return walk

def train_skipgram(walks, n_nodes, dim=128, lr=0.01, window=2, epochs=3):
    W1 = np.random.randn(n_nodes, dim)
    W2 = np.random.randn(dim, n_nodes)
    
    for epoch in range(epochs):
        for walk in walks:
            for i, target_node in enumerate(walk):
                # Node i -> Index i-1
                target_idx = target_node - 1 
                
                start_index = max(0, i - window)
                end_index = min(len(walk), i + window + 1)
                
                for j in range(start_index, end_index):
                    if i == j: continue 
                    
                    context_node = walk[j]
                    context_idx = context_node - 1 
                    
                    # Forward
                    h = W1[target_idx] 
                    scores = np.dot(W2.T, h)
                    
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / np.sum(exp_scores)
                    
                    # Backward
                    e = probs
                    e[context_idx] -= 1.0 
                    
                    dW2 = np.outer(h, e)
                    dh = np.dot(W2, e)
                    
                    W2 -= lr * dW2
                    W1[target_idx] -= lr * dh
                    
    return W1

def truncate_float(val):
    s = "{:.10f}".format(val)
    if '.' in s:
        head, tail = s.split('.')
        return head + "." + tail[:5]
    return s

def main():
    np.random.seed(1116)
    graph = Graph()
    edges = []

    if len(sys.argv) < 2:
        print("Usage: python hw3_3.py path/to/graph.txt")
        sys.exit(1)
        
    file_path = sys.argv[1]
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    edges.append((u, v))
                except ValueError: continue

    for edge in edges:
        graph.add_edge(*edge)

    # THe size of Matrix: Max ID (= V, assuming continuous 1..V)
    max_node_id = graph.get_max_node_id()
    n_nodes = max_node_id

    pq_pairs = [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)]
    target_nodes_to_print = [228, 102, 500, 73, 991]

    for p, q in pq_pairs:
        np.random.seed(1116) 
        
        # 1. Generate Deterministic Walks (Sorted nodes order)
        sorted_nodes = sorted(list(graph.nodes))
        walks = [node2vec_walk(graph, node, p=p, q=q) for node in sorted_nodes]
        
        # 2. Train (Size V, Index i-1)
        W1 = train_skipgram(walks, n_nodes, dim=128, lr=0.01, window=2, epochs=3)
        
        # 3. Print
        print(f"p:{p}, q:{q}")
        for target in target_nodes_to_print:
            #  Index i-1
            target_idx = target - 1
            if 0 <= target_idx < n_nodes:
                val = W1[target_idx][0]
                print(truncate_float(val))

if __name__ == "__main__":
    main()