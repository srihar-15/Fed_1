import copy
import torch
import numpy as np

class Server:
    def __init__(self, args, global_model):
        self.args = args
        self.global_model = global_model

    def aggregate(self, w):
        """
        FedAvg aggregation.
        w: list of state_dicts from clients
        """
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg
    
    def update_global_model(self, w_avg):
        self.global_model.load_state_dict(w_avg)

    def test(self, test_dataset, device):
        self.global_model.eval()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        return acc
