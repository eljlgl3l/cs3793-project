import threading

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

from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import base64

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

training_in_progress = False
current_epoch = 0
total_epochs = 0
train_results = None

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



def train_model(epochs=10):
    global best_val_acc, training_in_progress, current_epoch, total_epochs, train_results
    training_in_progress = True
    total_epochs = epochs
    current_epoch = 0

    # clear old results
    train_losses.clear()
    train_accuracies.clear()
    val_losses.clear()
    val_accuracies.clear()

    for i in range(epochs):
        current_epoch = i + 1

        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        train_results = (train_losses.copy(), val_losses.copy(), train_accuracies.copy(), val_accuracies.copy())

    training_in_progress = False
    train_results = (train_losses, val_losses, train_accuracies, val_accuracies)

    return train_losses, val_losses, train_accuracies, val_accuracies




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


# train_model_old()
#test_model()


app = Dash(__name__)

app.layout = html.Div([
    html.H1("Traffic Sign CNN Trainer"),
    html.Div([
        dcc.Input(id="epoch-input", type="number", value=5, min=1, step=1),
        html.Button("Train", id="train-button", n_clicks=0)
    ]),
    html.Div(id="status-text"),
    dcc.Graph(id="loss-graph"),
    dcc.Graph(id="acc-graph"),
    html.Img(id="png-output", style={"width": "600px"}),
    dcc.Interval(id="progress-interval", interval=1000, n_intervals=0),
    html.Div([
        html.Div(id="progress-bar", style={
            "width": "0%", "height": "30px", "backgroundColor": "green"
        })
    ], style={"width": "100%", "border": "1px solid #000"})
])

@app.callback(
    Output("status-text", "children"),
    Input("train-button", "n_clicks"),
    State("epoch-input", "value"),
    prevent_initial_call=True
)
def start_training(n_clicks, epochs):
    thread = threading.Thread(target=train_model, args=(epochs,))
    thread.start()
    return f"Training started for {epochs} epochs."


@app.callback(
    [Output("progress-bar", "style"),
     Output("loss-graph", "figure"),
     Output("acc-graph", "figure"),
     Output("png-output", "src")],
    Input("progress-interval", "n_intervals")
)
def update_progress(n):
    if total_epochs == 0:
        return {"width": "0%", "height": "30px", "backgroundColor": "green"}, go.Figure(), go.Figure(), None

    percent = int((current_epoch / total_epochs) * 100)
    bar_style = {"width": f"{percent}%", "height": "30px", "backgroundColor": "green"}

    if not training_in_progress and train_results:
        train_losses, val_losses, train_accs, val_accs = train_results

        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=list(range(1, len(train_losses) + 1)),
            y=train_losses, mode="lines+markers", name="Train Loss"
        ))
        loss_fig.add_trace(go.Scatter(
            x=list(range(1, len(val_losses) + 1)),
            y=val_losses, mode="lines+markers", name="Val Loss"
        ))
        loss_fig.update_layout(
            title="Loss per Epoch",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis=dict(range=[0, 5])  # adjust upper bound as needed
        )

        acc_fig = go.Figure()
        acc_fig.add_trace(go.Scatter(
            x=list(range(1, len(train_accs) + 1)),
            y=train_accs, mode="lines+markers", name="Train Acc"
        ))
        acc_fig.add_trace(go.Scatter(
            x=list(range(1, len(val_accs) + 1)),
            y=val_accs, mode="lines+markers", name="Val Acc"
        ))
        acc_fig.update_layout(
            title="Accuracy per Epoch",
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100])  # fixed 0–100%
        )

        encoded = base64.b64encode(open("predictions.png", "rb").read()).decode()
        img_src = "data:image/png;base64," + encoded

        return bar_style, loss_fig, acc_fig, img_src

    return bar_style, go.Figure(), go.Figure(), None




fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.ravel()

random_indices = np.random.choice(len(full_dataset), 16, replace=False)

# with torch.no_grad():
#     for i, idx in enumerate(random_indices):
#         img, label = full_dataset[idx]
#         output = model(img.unsqueeze(0))
#         maximum, pred = torch.max(output, 1)
#
#         # Display image
#         img_show = img.permute(1, 2, 0).numpy()
#         axes[i].imshow(img_show)
#         axes[i].set_title(f'True: {full_dataset.classes[label]}\nPred: {full_dataset.classes[pred.item()]}')
#         axes[i].axis('off')
#
# plt.tight_layout()
# plt.savefig('predictions.png')
# plt.show()



if __name__ == "__main__":
    app.run(debug=True)
