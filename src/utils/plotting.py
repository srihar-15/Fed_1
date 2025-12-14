import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(loss_train, filename):
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.savefig(filename)
    plt.close()

def plot_accuracy_curve(acc_test, filename):
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test_accuracy')
    plt.xlabel('epochs')
    plt.savefig(filename)
    plt.close()

def plot_client_distribution(dict_users, dataset_targets, filename):
    """
    Plot the distribution of labels for each client.
    """
    num_clients = len(dict_users)
    num_classes = 10
    
    # Check if dataset_targets is a tensor or list
    if isinstance(dataset_targets, list):
        targets_np = np.array(dataset_targets)
    else:
        targets_np = dataset_targets.numpy()

    data_map = np.zeros((num_clients, num_classes))
    
    for i in range(num_clients):
        client_indices = list(dict_users[i])
        client_labels = targets_np[client_indices]
        counts = np.bincount(client_labels, minlength=num_classes)
        data_map[i] = counts

    plt.figure(figsize=(10, 8))
    plt.imshow(data_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.ylabel('Client ID')
    plt.xlabel('Class Label')
    plt.title('Label Distribution per Client')
    plt.savefig(filename)
    plt.close()
