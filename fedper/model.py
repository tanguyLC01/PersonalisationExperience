
import torch
import torch.nn as nn
from typing import Tuple, List

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(net, trainloader, epochs: int, verbose=False) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
            
def test(net, testloader) -> Tuple[float, float]:  
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(DEVICE), batch['label'].to(DEVICE)
            outputs = net(images)
            
            # Metrics
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# We use the architecture of the McMahan paper for the global network
class PersonalizedNet(nn.Module):
    """Entire network for personalized federated learning."""
    
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.global_net = Global_Net(model_type)
        self.local_net = Local_Net(num_classes, model_type)
        
    def forward(self, x):
        x = self.global_net(x)
        x = self.local_net(x)
        return x
    
class Global_Net(nn.Module):
    
    def __init__(self, model_type="globalier"):
        super().__init__()
        self.model_type = model_type
        self.model = nn.Sequential(nn.Conv2d(1, 32, 5), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        if self.model_type == "globalier":
            self.model.add_module("conv2", nn.Sequential(nn.Conv2d(32, 64, 5), 
                                  nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten()))
            self.model.add_module("fc1", nn.Sequential(nn.Linear(64 * 16, 512),
                                   nn.ReLU()))
        
    def forward(self, x):
        return self.model(x)
        
class Local_Net(nn.Module):
    
    def __init__(self, num_classes, model_type='globalier'):
        super().__init__()
        self.model_type = model_type
        self.model = nn.Sequential()
        if self.model_type == 'localier':
            self.model.add_module("conv2", nn.Sequential(nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten()))
            self.model.add_module("fc1", nn.Sequential(nn.Linear(64 * 16, 512), nn.ReLU()))
            
        self.model.add_module('fc2', nn.Linear(512, num_classes))
        
    def forward(self, x):
        return self.model(x)    
