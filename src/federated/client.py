import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class ClientDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Client:
    def __init__(self, args, dataset, idxs, device):
        self.args = args
        self.loader = DataLoader(ClientDataset(dataset, idxs), batch_size=args.local_bs, shuffle=True)
        self.device = device

    def train(self, model):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.loader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return model.state_dict(), sum(epoch_loss)/len(epoch_loss)
