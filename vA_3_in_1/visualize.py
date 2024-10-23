import subprocess
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# List of number of layers to experiment with
num_layers_list = [2, 4, 8, 16, 32, 64, 128]

# Methods to test
methods = ['pmlp_gcn', 'gcn', 'mlp']
method_labels = {'pmlp_gcn': 'PMLP-GCN', 'gcn': 'GCN', 'mlp': 'MLP'}
method_colors = {'pmlp_gcn': 'red', 'gcn': 'blue', 'mlp': 'orange'}
method_markers = {'pmlp_gcn': 's', 'gcn': 'o', 'mlp': '^'}

# Initialize results dictionary
results = {method: [] for method in methods}

# Other hyperparameters (you can adjust these as needed)
lr = 0.1
weight_decay = 0.01
dropout = 0.5
hidden_channels = 64

# Loop over methods and number of layers
for method in methods:
    print(f"Running experiments for method: {method}")
    for num_layers in num_layers_list:
        # Construct the command to run main.py with the current number of layers and method
        cmd = [
            sys.executable,  # Path to the Python interpreter
            'main.py',
            '--dataset', 'pubmed',  # Adjust the dataset as needed
            '--method', method,
            '--protocol', 'semi',
            '--lr', str(lr),
            '--weight_decay', str(weight_decay),
            '--dropout', str(dropout),
            '--num_layers', str(num_layers),
            '--hidden_channels', str(hidden_channels),
            '--induc',
            '--runs', '1',
            '--epochs', '1',
            '--device', '0',
            '--conv_tr',
            '--conv_va',
            '--conv_te'
        ]

        print(f"Running with num_layers={num_layers}")
        try:
            # Run main.py as a subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse the output
            mean_performance = None
            for line in result.stdout.strip().split('\n'):
                try:
                    output = json.loads(line)
                    mean_performance = output['mean_performance']
                    mean_performance = float(mean_performance)
                    std_performance = output['std_performance']
                    break
                except json.JSONDecodeError:
                    continue  # Not the JSON line we're looking for

            if mean_performance is not None:
                # Store the result
                results[method].append(mean_performance)
                print(f"Mean Performance: {mean_performance}")
            else:
                print("Failed to parse mean performance from output.")
                results[method].append(None)

        except subprocess.CalledProcessError as e:
            print(f"An exception occurred while running main.py: {e.stderr}")
            results[method].append(None)

# Prepare data for plotting
layers_indices = list(range(len(num_layers_list)))
x_labels = num_layers_list

# Plotting the data
plt.figure(figsize=(6, 4))

for method in methods:
    accuracies = results[method]
    # Replace None with 0 for plotting
    accuracies = [acc if acc is not None else 0 for acc in accuracies]
    plt.plot(
        layers_indices,
        accuracies,
        marker=method_markers[method],
        color=method_colors[method],
        label=method_labels[method]
    )

plt.xlabel('Number of layers')
plt.ylabel('Accuracy')
plt.title('Pubmed')

# Update the x-axis ticks and labels
plt.xticks(layers_indices, x_labels)
plt.yticks(np.arange(20, 81, 10))

plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Optionally, save the plot
# plt.savefig('accuracy_vs_layers.png')

# Print the results
print("\nResults:")
for method in methods:
    print(f"Method: {method_labels[method]}")
    for num_layers, acc in zip(x_labels, results[method]):
        print(f"  Number of Layers: {num_layers}, Mean Accuracy: {acc}")
