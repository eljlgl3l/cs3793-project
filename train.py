import matplotlib.pyplot as plt
import numpy as np
from lisa import LISA
import torch, torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import random

# Load dataset
full_dataset = LISA(root='./data', train=True, download=False, transform=None)

train_size = (int(.7 * len(full_dataset)))
test_size = (int(.15 * len(full_dataset)))
val_size = len(full_dataset) - train_size - test_size

train_dataset, test_dataset, val_dataset = random_split(full_dataset, [train_size, test_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(full_dataset.classes)

# Define CNN model
class NeuralNetwork (torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # convolution layer, kernel size 3
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # max pooling layer, 2x2 feature map
        self.pool = torch.nn.MaxPool2d(2, 2)
        # fully connected layer, maps 32*16*16 feature map to 256 features
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        # output layer, maps 256 features to number of classes
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # flatten feature map 
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NeuralNetwork()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_acc = 0.0

def train_model ():
    global best_val_acc

    # Training loop - loops for number of times we iterate through the dataset
    for i in range(35):
        running_loss = 0.0
        correct = 0
        total = 0

        # loop through data in batches
        for j, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # get model predictions
            outputs = model(inputs)
            # calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # total loss
            running_loss += loss.item()

            # calculate accuracy
            values, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                values, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f'Epoch {i + 1}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.2f}%, Val loss: {val_loss:.3f}, Val acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print (f'saved best model with val acc: {best_val_acc:.2f}%')

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()


def test_model ():
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            values, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f'Final test accuracy: {test_acc:.2f}%')


train_model()
test_model()

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.ravel()

random_indices = np.random.choice(len(full_dataset), 16, replace=False)

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        img, label = full_dataset[idx]
        output = model(img.unsqueeze(0))
        maximum, pred = torch.max(output, 1)
        
        # Display image
        img_show = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img_show)
        axes[i].set_title(f'True: {full_dataset.classes[label]}\nPred: {full_dataset.classes[pred.item()]}')
        axes[i].axis('off')

plt.tight_layout()
plt.savefig('predictions.png')
plt.show()