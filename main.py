import copy
import time
import numpy as np
import torch
from multiprocessing import Pool, set_start_method

from src.utils.options import args_parser
from src.utils.plotting import plot_accuracy_curve, plot_client_distribution
from src.data.partition import get_dataset, iid_partition, non_iid_partition
from src.models.cnn import SimpleCNN
from src.federated.client import Client
from src.federated.server import Server

def client_update(args_tuple):
    """
    Standalone function for multiprocessing.
    """
    args, dataset, idxs, model_state, device = args_tuple
    # Re-instantiate model to avoid pickling the entire object if possible, 
    # or just load state_dict.
    model = SimpleCNN(args.num_classes).to(device)
    model.load_state_dict(model_state)
    
    client = Client(args, dataset, idxs, device)
    w, loss = client.train(model)
    return w, loss

if __name__ == '__main__':
    # Set start method to spawn for better compatibility
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f"Using device: {args.device}")

    # Load Dataset
    train_dataset, test_dataset = get_dataset()

    # Partition Dataset
    if args.iid:
        dict_users = iid_partition(train_dataset, args.num_users)
        plot_name = f'results/iid_distribution.png'
    else:
        dict_users = non_iid_partition(train_dataset, args.num_users, args.alpha)
        plot_name = f'results/non_iid_alpha_{args.alpha}_distribution.png'
    
    # Save distribution plot
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plot_client_distribution(dict_users, train_dataset.targets, plot_name)

    # Initialize Global Model
    global_model = SimpleCNN(args.num_classes).to(args.device)
    global_model.train()
    print(global_model)

    # Copy weights
    global_weights = global_model.state_dict()

    # Server
    server = Server(args, global_model)

    # Training Loop
    train_accuracy = []
    
    print(f"Starting Training: IID={args.iid}, Alpha={args.alpha}, Clients={args.num_users}")
    
    # Multiprocessing setup
    # Note: On Windows, 'spawn' is default. We need to be careful with passing heavy objects like dataset.
    # In a real heavy dataset scenario, we wouldn't pass the whole dataset copy. 
    # CIFAR10 is small enough to pass or re-load, but passing torch Dataset via MP can be slow.
    # We will use serial execution if args.gpu != -1 (GPU sharing across processes is tricky without complex setup),
    # Or just sequential for simplicity unless required. user requested multiprocessing.
    # Let's try multiprocessing but fallback to serial if it causes issues or if dataset pickling is too slow.
    
    use_mp = True # Toggle for debugging
    
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |')
        
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # Prepare args for clients
        client_args = []
        for idx in idxs_users:
            client_args.append((args, train_dataset, dict_users[idx], copy.deepcopy(global_weights), args.device))
        
        if use_mp and args.device.type == 'cpu': 
            # Only use MP on CPU to avoid CUDA re-initialization error in forked processes
            # Also, passing dataset might be slow.
            with Pool(processes=min(m, 4)) as pool: # Limit pool size
                results = pool.map(client_update, client_args)
            
            for w, loss in results:
                local_weights.append(copy.deepcopy(w))
                local_losses.append(loss)
        else:
            # Serial execution
            for idx in idxs_users:
                client = Client(args, train_dataset, dict_users[idx], args.device)
                model_copy = copy.deepcopy(global_model) # Use current global model
                model_copy.load_state_dict(global_weights) # Ensure it has latest weights
                w, loss = client.train(model_copy)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(loss)
        
        # Aggregate
        global_weights = server.aggregate(local_weights)
        server.update_global_model(global_weights)
        
        # Test accuracy
        acc = server.test(test_dataset, args.device)
        train_accuracy.append(acc)
        print(f'Round {epoch+1}, Test Accuracy: {acc:.2f}%')

    # Save Results
    file_name = f'results/fedavg_iid{args.iid}_alpha{args.alpha}_C{args.frac}'
    np.save(f'{file_name}_accuracy.npy', np.array(train_accuracy))
    plot_accuracy_curve(train_accuracy, f'{file_name}_accuracy.png')
    print("Training Complete. Results saved.")
