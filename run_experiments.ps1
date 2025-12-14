# Short demo run (2 rounds each) to verify pipeline and generate plots.
# User should increase epochs to 50 for full analysis.

$env:PYTHONPATH = "."

# Centralized (1 client, 2 epochs)
Write-Host "Running Centralized Baseline..."
python main.py --epochs 2 --num_users 1 --frac 1.0 --iid --local_ep 1 --dataset cifar --gpu -1

# IID (10 clients, 2 rounds)
Write-Host "Running IID..."
python main.py --epochs 2 --num_users 10 --frac 1.0 --iid --local_ep 1 --dataset cifar --gpu -1

# Non-IID
Write-Host "Running Non-IID Alpha=1.0..."
python main.py --epochs 2 --num_users 10 --frac 1.0 --alpha 1.0 --local_ep 1 --dataset cifar --gpu -1

Write-Host "Running Non-IID Alpha=0.5..."
python main.py --epochs 2 --num_users 10 --frac 1.0 --alpha 0.5 --local_ep 1 --dataset cifar --gpu -1

Write-Host "Running Non-IID Alpha=0.1..."
python main.py --epochs 2 --num_users 10 --frac 1.0 --alpha 0.1 --local_ep 1 --dataset cifar --gpu -1

# Plot
Write-Host "Generating Comparison..."
python src/utils/plot_comparison.py

Write-Host "Done."
