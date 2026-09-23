import numpy as np
import ast

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

num_layers = int(input("Enter number of layers (including input and output): "))
layer_sizes = ast.literal_eval(input("Enter number of nodes in each layer as a list (e.g., [2, 2, 1]): "))
target_y = np.array(ast.literal_eval(input("Enter target actual output of y as a 2D list (e.g., [[0.5]]): ")))
error_threshold = float(input("Enter acceptable difference threshold for y (e.g., 0.001): "))
lr = float(input("Enter learning rate: "))
X = np.array(ast.literal_eval(input("Enter input values as a 2D list (e.g., [[0.35, 0.9]]): ")))

weights = []
for i in range(num_layers - 1):
    w_shape = (layer_sizes[i], layer_sizes[i+1])
    w = np.array(ast.literal_eval(input(f"Enter weights between layer {i+1} and {i+2} as a 2D list of shape {w_shape}: ")))
    weights.append(w)

biases = []
for i in range(num_layers - 1):
    b_shape = (1, layer_sizes[i+1])
    b_input = input(f"Enter biases for layer {i+2} as a 2D list of shape {b_shape} (or press Enter for zeros): ")
    if b_input.strip() == "":
        b = np.zeros(b_shape)
    else:
        b = np.array(ast.literal_eval(b_input))
    biases.append(b)

error = float('inf')
iteration = 0

while error > error_threshold:
    activations = [X]
    for i in range(num_layers - 1):
        net_input = np.dot(activations[-1], weights[i]) + biases[i]
        activations.append(sigmoid(net_input))
    
    error = np.max(np.abs(target_y - activations[-1]))
    
    if error <= error_threshold:
        break
        
    deltas = [(activations[-1] - target_y) * sigmoid_derivative(activations[-1])]
    
    for i in range(num_layers - 2, 0, -1):
        delta = np.dot(deltas[-1], weights[i].T) * sigmoid_derivative(activations[i])
        deltas.append(delta)
        
    deltas.reverse()
    
    for i in range(num_layers - 1):
        weights[i] -= lr * np.dot(activations[i].T, deltas[i])
        biases[i] -= lr * np.sum(deltas[i], axis=0, keepdims=True)
    
    iteration += 1

print(f"Converged in {iteration} iterations.")
print("Final Difference:", error)
print("Final Weights:", weights)
print("Final Outputs:", activations[-1])