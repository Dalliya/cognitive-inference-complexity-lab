import sys
import os
import numpy as np

# Adjust system path to allow imports from core and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.neural_network import neuralNetwork
from utils.visualizer import save_3d_topology

# 1. Initialize Network Architecture
# 784 inputs (28x28 pixels), 200 hidden nodes, 10 outputs (0-9 digits), 0.1 Learning Rate
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
epochs = 5

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
labels = [str(i) for i in range(10)]

print(f"[EXECUTION] Phase 1: Training on Full MNIST Dataset ({epochs} Epochs)...")

# 2. Training Phase (Full Dataset)
# Ensure 'mnist_train.csv' is present in the 'data/mnist/' directory
with open("data/mnist/mnist_train.csv", 'r') as f:
    training_data = f.readlines()

for e in range(epochs):
    print(f" > Processing Epoch {e+1} of {epochs}...")
    
    # SKIP THE FIRST ROW (HEADER) USING [1:]
    for record in training_data[1:]:
        vals = record.split(',')
        # Using np.asarray(..., dtype=float) for compatibility with NumPy 2.0+
        inputs = (np.asarray(vals[1:], dtype=float) / 255.0 * 0.99) + 0.01
        
        # Create target array (0.01 for all, 0.99 for the correct label)
        targets = np.zeros(output_nodes) + 0.01
        targets[int(vals[0])] = 0.99
        n.train(inputs, targets)

print("[EXECUTION] Phase 2: Testing and generating Matrix Topology...")

# 3. Testing Phase (Full Dataset)
conf_matrix = np.zeros((10, 10))
scorecard = []

# Ensure 'mnist_test.csv' is present in the 'data/mnist/' directory
with open("data/mnist/mnist_test.csv", 'r') as f:
    test_data = f.readlines()

# SKIP THE FIRST ROW (HEADER) USING [1:]
for record in test_data[1:]:
    vals = record.split(',')
    target = int(vals[0])
    inputs = (np.asarray(vals[1:], dtype=float) / 255.0 * 0.99) + 0.01
    
    # Query the network for prediction
    outputs = n.query(inputs)
    prediction = np.argmax(outputs)
    
    # Update metrics and confusion matrix
    conf_matrix[target, prediction] += 1
    scorecard.append(1 if prediction == target else 0)

# 4. Calculate Final Metrics and Export Visualization
acc = (sum(scorecard) / len(scorecard)) * 100
print(f"[METRICS] Final Model Accuracy: {acc:.2f}%")

# Save and show the 3D topology
save_3d_topology(conf_matrix, labels, acc, "mnist_full_topology.png")