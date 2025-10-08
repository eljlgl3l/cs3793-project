import matplotlib.pyplot as plt
import numpy as np
from lisa import LISA
import torch, torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Load dataset
dataset = LISA(root='./data', train=True, download=False, transform=None)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

num_classes = len(dataset.classes)

class NeuralNetwork (torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 16 * 16, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = NeuralNetwork()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)


for i in range(20):
    running_loss = 0.0
    correct = 0
    total = 0

    for j, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        values, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if j % 50 == 49:
            print('[%d, %5d] loss: %.3f | accuracy: %.2f%%' %
                  (i + 1, j + 1, running_loss / 50, 100 * correct / total))
            running_loss = 0.0
