# Federated Learning from Scratch (Non-IID Analysis)

This project implements FedAvg manually to analyze the impact of Non-IID data distributions on model performance.

## Structure
- `src/data/partition.py`: Implements Dirichlet distribution for Non-IID splits.
- `src/models/cnn.py`: Simple CNN for CIFAR-10.
- `src/federated/`: Contains `Client` (local training) and `Server` (aggregation) classes.
- `main.py`: Orchestrates the simulation with `multiprocessing`.

## Tech Stack
- PyTorch
- Python Multiprocessing (simulating clients)
- CIFAR-10 Dataset

## Experiments
Run `run_experiments.ps1` to execute the full suite:
1.  **Centralized**: Baseline performance.
2.  **IID**: Data uniformly distributed.
3.  **Non-IID (α=1.0)**: Moderate heterogeneity.
4.  **Non-IID (α=0.1)**: High heterogeneity (most clients have only 1-2 classes).

## Why FedAvg Breaks on Non-IID Data
When $\alpha$ is low (e.g., 0.1), clients possess highly skewed label distributions (e.g., Client A has only "Dog" images, Client B has only "Cat").
- **Client Drift**: Local training pushes Client A's model to predict "Dog" for everything, and Client B's to predict "Cat".
- **Weight Divergence**: The weights $w_A$ and $w_B$ move in different directions in the loss landscape.
- **Aggregation Failure**: Averaging these divergent weights ($\frac{w_A + w_B}{2}$) results in a global model that performs poorly on both, often worse than IID training where updates are more aligned.

## Results
Results are saved in `results/` folder:
- `*_accuracy.png`: Accuracy curves.
- `*_distribution.png`: Visualizes how classes are split among clients.
- `convergence_comparison.png`: Overlaid plot of all experiments.
