import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

X_train = torch.load('X_train_combo-326.pt')
combo_labels_train = torch.load('combo_labels_train-326.pt')
print(X_train.shape)
print(combo_labels_train.shape)
X_test = torch.load('X_test_combo-326.pt')
combo_labels_test = torch.load('combo_labels_test-326.pt')


train_dataset = TensorDataset(X_train, combo_labels_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test, combo_labels_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResNet, self).__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layers = self.make_layers()
        self.fc = nn.Linear(64, output_size)

    def make_layers(self):
        layers = []
        layers.append(ResidualBlock(64, 64))
        layers.append(ResidualBlock(64, 64))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layers(out)
        out = out.mean(2)
        out = self.fc(out)
        return out

best_accuracy = 0.0
patience_counter = 0
train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []
test_loss_list = []

model = ResNet(input_size=125, output_size=48)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    correct_test = 0
    total_train = 0
    for  inputs, combo_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, combo_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_train += combo_labels.size(0)
        correct_train += (predicted == combo_labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
    print(f"    Train Accuracy: {train_accuracy}%")

    # Evaluation
    model.eval()
    direction_correct = 0
    action_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, combo_labels in test_loader:
            outputs = model(inputs)
            test_loss = criterion(outputs, combo_labels)
            _, predicted = torch.max(outputs, 1)

            total += combo_labels.size(0)
            correct_test += (predicted == combo_labels).sum().item()

    test_accuracy = 100 * correct_test / total
    print(f"    Test Accuracy: {test_accuracy}%")
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy

        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= 10:
        print("Early stopping")
        break
    train_loss_list.append(total_loss / len(train_loader))
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)
    test_loss_list.append(test_loss / len(train_loader))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.title('Accuracy vs. epochs', fontsize=22)
plt.xlabel('Epochs', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.legend(prop={'size': 22})

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_list, label='Train Accuracy')
plt.plot(test_accuracy_list, label='Test Accuracy')
plt.title('Accuracy vs. epochs', fontsize=22)
plt.xlabel('Epochs', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.legend(prop={'size': 22})

plt.tight_layout()
plt.show()