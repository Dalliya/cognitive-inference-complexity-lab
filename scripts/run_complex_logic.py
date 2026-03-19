import sys
import os
import numpy as np

# Adjust system path to allow imports from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.neural_network import neuralNetwork
from utils.visualizer import save_3d_topology

# 1. Initialize Network Architecture
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
epochs = 5

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Text labels for Fashion-MNIST categories
fashion_labels = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot"
]

print(f"[EXECUTION] Phase 1: Training on Full Fashion-MNIST Dataset ({epochs} Epochs)...")

# 2. Training Phase (Using the full dataset)
# Ensure the files are located in the data/fashion/ directory
with open("data/fashion/fashion-mnist_train.csv", 'r') as f:
    training_data = f.readlines()

for e in range(epochs):
    print(f" > Processing Epoch {e+1} of {epochs}...")
    
    # Skip the first row (header) using the [1:] slice
    for record in training_data[1:]:
        vals = record.split(',')
        inputs = (np.asarray(vals[1:], dtype=float) / 255.0 * 0.99) + 0.01
        
        targets = np.zeros(output_nodes) + 0.01
        targets[int(vals[0])] = 0.99
        n.train(inputs, targets)

print("[EXECUTION] Phase 2: Testing and generating Matrix Topology...")

# 3. Testing Phase
conf_matrix = np.zeros((10, 10))
scorecard = []

with open("data/fashion/fashion-mnist_test.csv", 'r') as f:
    test_data = f.readlines()

# Skip the first row (header) using the [1:] slice
for record in test_data[1:]:
    vals = record.split(',')
    target = int(vals[0])
    inputs = (np.asarray(vals[1:], dtype=float) / 255.0 * 0.99) + 0.01
    
    outputs = n.query(inputs)
    prediction = np.argmax(outputs)
    
    conf_matrix[target, prediction] += 1
    scorecard.append(1 if prediction == target else 0)

# 4. Calculate Metrics and Export Visualization
acc = (sum(scorecard) / len(scorecard)) * 100
print(f"[METRICS] Final Model Accuracy: {acc:.2f}%")

# Call the visualization utility to generate the 3D topology
save_3d_topology(conf_matrix, fashion_labels, acc, "fashion_full_topology.png")