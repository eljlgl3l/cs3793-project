import matplotlib.pyplot as plt
import numpy as np
from lisa import LISA
import torch, torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

# Load dataset
dataset = LISA(root='./data', train=True, download=False, transform=None)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

num_classes = len(dataset.classes)

# Define CNN model
class NeuralNetwork (torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # convolution layer, kernel size 3
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # max pooling layer, 2x2 feature map
        self.pool = torch.nn.MaxPool2d(2, 2)
        # fully connected layer, maps 32*16*16 feature map to 256 features
        self.fc1 = torch.nn.Linear(32 * 16 * 16, 256)
        # output layer, maps 256 features to number of classes
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        # flatten feature map 
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NeuralNetwork()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)


# Training loop - loops for number of times we iterate through the dataset
for i in range(20):
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

        # print progress every 50 batches
        if j % 50 == 49:
            print('[%d, %5d] loss: %.3f | accuracy: %.2f%%' %
                  (i + 1, j + 1, running_loss / 50, 100 * correct / total))
            running_loss = 0.0

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.ravel()

random_indices = np.random.choice(len(dataset), 16, replace=False)

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        img, label = dataset[idx]
        output = model(img.unsqueeze(0))
        maximum, pred = torch.max(output, 1)
        
        # Display image
        img_show = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img_show)
        axes[i].set_title(f'True: {dataset.classes[label]}\nPred: {dataset.classes[pred.item()]}')
        axes[i].axis('off')

plt.tight_layout()
plt.savefig('predictions.png')
plt.show()