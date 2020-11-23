import matplotlib.pyplot as plt
import numpy as np

from queue import PriorityQueue

import argparse
parser = argparse.ArgumentParser() #initialize argument parser


# Parse arguments from the command line
parser.add_argument('-data', default=f'data1.txt') 
parser.add_argument('-k', default=20) 
parser.add_argument('-s', default=5) 
parser.add_argument('-d', default=0.9)
parser.add_argument('-output', default='output1.txt') 

args = parser.parse_args()

# User specified arguments
data_path = str(args.data) # -data
max_cluster_num = int(args.k) # -k
min_cluster_size = int(args.s) # -s
min_intra_dist = int(args.d) # -d
output_path = str(args.output) # -output

# Import dataset
data = np.genfromtxt(data_path, delimiter=' ')

DATA_SIZE = data.shape[0]
DATA_DIM = data.shape[1]
default_width = np.max(data)-np.min(data)

# We will access data by their index
feature_idx = np.array([i for i in range(DATA_SIZE)], dtype=int)

# Initialize labels
label = {i: 0 for i in range(DATA_SIZE)}

# Wrapper function for computing similarities
def dist(a, b):
    return np.linalg.norm(a - b)

# Function to calculate intra-cluster distance
def icd(features, center):

    intra_dist = np.array([])

    for f in features:
        
        # Find similarities between each data point in the cluster to its center
        distance = dist(center, f)
        intra_dist = np.append(intra_dist, distance)

    return np.mean(intra_dist)

# The K-Means Algorithm
def kmeans(feature_idx, k=2):

    feature_dim = DATA_DIM
    
    # Initialize K centers
    center_idx = np.random.choice(feature_idx, size=k, replace=False)
    center = np.array(data[center_idx])
    cluster_id = [i for i in range(k)]

    # Initialize return values
    width = [] # intra-cluster distance
    clusters = []

    while True:

        # Assign each x to its closest center
        label = np.array([], dtype=int)
        dist2center = np.array([])

        for f in feature_idx:

            # Initialize variable for finding the closest center
            min_distance = None
            nearest_label = None

            # Find distance between a data point to all cluster centers
            for i in range(k):
                
                c = center[i]
                distance = dist(data[f], c)

                # Update to the closest center so far
                if min_distance is None or distance < min_distance:  # Short circuit
                    
                    min_distance = distance
                    nearest_label = cluster_id[i]
            
            # Record the distance to cluster center and their labels for each data point in this iteration
            dist2center= np.append(dist2center, min_distance)
            label = np.append(label, nearest_label)

        # Calculate the mean of each cluster
        cluster_mean = np.array([])

        for lab in cluster_id:

            cluster = np.array([data[i] for i in feature_idx[label == lab]])
            cluster_mean = np.append(cluster_mean, np.mean(cluster, axis=0))

        cluster_mean = np.reshape(cluster_mean, (-1, feature_dim))

        # Continue updating the cluster centers if any x is reassigned to a new center
        if np.array_equal(cluster_mean, center):
            
            # Output block:
            # Calculate the width of each cluster
            bitmap = []
            
            # Create a bitmap to filter data points only in this cluster
            for i, lab in zip(range(k), cluster_id):
                
                idx_position = label == lab
                bitmap.append((sum(idx_position), idx_position))
            
            # Order the cluster from the smallest to the largest
            bitmap.sort(key=lambda x : x[0])
            
            for (_, idx_position) in bitmap:
                
                width.append(np.mean(dist2center[idx_position]))
                clusters.append(feature_idx[idx_position])
            
            break

        else:
            
            # Update the centers
            center = cluster_mean

    return width, clusters

# Data structure for the dendrogram
class Node:
    
    # Interally determine the current leaves with a Priority Queue
    node_queue = PriorityQueue()
    priority_set = set()
    
    def __init__(self, feature_index, width=default_width):
        
        # Data and children
        self.data = feature_index
        self.left = None
        self.right = None
        
        # Attributes
        self.size = len(feature_index)
        self.width = width
        self.isLeaf = False
        self.end = False
        
        self.enqueue(offset=self.size)

    def clear_node_queue():
        """
        Helper function to clear the interal priority queue
        """
        
        Node.node_queue = PriorityQueue()
    
    def dequeue():
        """
        Dequeue the node_queue from the front, 
        i.e. the node with the largest number of data points so far
        """
        
        return Node.node_queue.get()[1]
    
    def qsize():
        """
        Return the length of node_queue
        """
        
        return Node.node_queue.qsize()
    
    def queue_is_empty():
        """
        Return boolean if node_queue is empty
        """
        
        return Node.node_queue.empty()
    
    def enqueue(self, offset=0):
        """
        Enqueue to node_queue, offset by a position number to avoid duplicated ordering keys
        """
        
        priority = Node.generate_priority(DATA_SIZE - offset)
        Node.node_queue.put((priority, self))
        Node.priority_set.add(priority)
    
    def generate_priority(priority):
        """
        Helper function to generate a unique ordering key for each insertion to the node_queue
        """
        
        if not {priority}.issubset(Node.priority_set):
        
            return priority
        
        else:
            
            # If there is a duplication, try the next available position
            return Node.generate_priority(priority + 1 / DATA_SIZE)

# Recursively call itself to build the dendrogram
def bisecting(recursive=True):
    
    node = Node.dequeue()
    
    if node.isLeaf:
        
        node.enqueue()
        return True # Stop
    
    # Stopping criteria
    if node.width < min_intra_dist:
        
        node.isLeaf = True
        node.enqueue()
        
        return bisecting() if recursive else False # Continue to next node
    
    # Stopping criteria    
    if node.size < min_cluster_size:
        
        node.enqueue()
        return True # Stop
    
    # Bisecting the current cluster into two smaller clusters
    [left_width, right_width], [left_cluster, right_cluster] = kmeans(node.data)
    
    # Put them into our dendrogram
    node.left = Node(left_cluster, left_width)
    node.right = Node(right_cluster, right_width)
    
    # Stopping criteria   
    if Node.qsize() >= max_cluster_num:
        
        return True # Stop
    
    return bisecting() if recursive else False # Continue to next node

# A helper class for storing the nodes at each depth
class store:
    
    # Storage
    storage = {}

    def emit(depth, data):

        # Add data to the current depth
        current = store.storage.get(depth, [])
        current.append(str(data))
        store.storage[depth] = [i for i in current]
        
    def print_tree():

        # Print node size at each depth
        for k in range(len(store.storage)):
            if store.storage[k]:
                line = ' '.join(store.storage[k])
                print(line)

# Print dendrogram as required by the homework instruction
def print_dendrogram(node, depth=0):
    
    line = ''
    padding = depth
    
    if not node:
        return
    
    # Pre-order traversal: Root -> Left -> Right
    store.emit(depth, node.size)
    print_dendrogram(node.left, padding+1)
    print_dendrogram(node.right, padding+1)
    
    return

# A better way to print the tree (vertically)
def print_dendrogram_2(node, depth=1, pos=[1], right_tree=True, print_icd=False):
    
    line = ''
    padding = depth
    position = [pos[i] for i in range(len(pos))]
    
    if not node:
        return
    
    # For drawing branches
    for i in range(padding-1):
        if position[i] != 1:
            line += '│     '
        else:
            line += '      '
    
    if right_tree:
        line += '└──── '
    else:
        line += '├──── '
        
    
    line += str(node.size)
    
    # Identifying leaf nodes
    if not node.left or not node.right:
        
        leaf = f'icd = {node.width:.3f}' if print_icd else ''
        line += ': Leaf ' + leaf
    
    # Pre-order traversal
    print(line)
    
    # For drawing branches
    position_l = [pos[i] for i in range(len(pos))]
    position_l.append(0)

    print_dendrogram_2(node.left, padding+1, position_l, False, print_icd)
    
    # For drawing branches
    position_r = [pos[i] for i in range(len(pos))]
    position_r.append(1)

    print_dendrogram_2(node.right, padding+1, position_r, True, print_icd)

# Initialize the root node with all the data
Node.clear_node_queue()
root = Node(feature_idx)

# Run the bisecting algorithm
bisecting()

print()
# Basic dendrogram
print('Dendrogram:')
store.storage = {i:[] for i in range(8)}
print_dendrogram(root)
store.print_tree()

print()
# Better looking dendrogram
print('A better looking dendrogram:')
print(Node.qsize(), 'leaves')
print_dendrogram_2(root)

# Label the clusters from left to right
def print_clusters(node, leaf_visited):
    
    if node.isLeaf:
        
        # Update its label
        for i in node.data:
            label[i] = leaf_visited
        
        return leaf_visited + 1
    
    # Post-order traversal - depth-first search
    if node.left:
        
        leaf_visited = print_clusters(node.left, leaf_visited)
    
    if node.right:
        
        leaf_visited = print_clusters(node.right, leaf_visited)
    
    # Return the total leaf node count
    return leaf_visited

# Set the indicator for all leaf nodes (which are all in the node_queue now)
while not Node.queue_is_empty():

    node = Node.dequeue()
    node.isLeaf = True

# Update label dictionary
total_leaf_node = print_clusters(root, leaf_visited=0)

# Output to file
label_output = [v for _,v in label.items()]
np.savetxt(output_path, label_output, delimiter=' ', fmt='%i')

