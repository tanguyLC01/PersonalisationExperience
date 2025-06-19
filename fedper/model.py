
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
    """Global network for personalized federated learning."""
    
    def __init__(self, num_classes):
        super().__init__()
        self.global_net = Global_Net()
        self.local_net = Local_Net(num_classes)
        
    def forward(self, x):
        x = self.global_net(x)
        x = self.local_net(x)
        return x
    
class Global_Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        return x
        
class Local_Net(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x     
