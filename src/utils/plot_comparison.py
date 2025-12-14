import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_comparison():
    results_dir = 'results'
    files = glob.glob(os.path.join(results_dir, '*_accuracy.npy'))
    
    plt.figure(figsize=(10, 6))
    
    for f in files:
        label = os.path.basename(f).replace('_accuracy.npy', '').replace('fedavg_', '')
        acc = np.load(f)
        plt.plot(range(len(acc)), acc, label=label)
    
    plt.xlabel('Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Federated Learning Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/convergence_comparison.png')
    print("Comparison plot saved to results/convergence_comparison.png")

if __name__ == "__main__":
    plot_comparison()
