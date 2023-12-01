#use CNN for actions and diretions
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from amp_time_functionzidian_success import amp_time
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

X_train = torch.load('X_train_combo-326.pt')
combo_labels_train = torch.load('combo_labels_train-326.pt')
print(X_train.shape)
print(combo_labels_train.shape)
# X_test = torch.load('X_train_combo-127.pt')
# combo_labels_test = torch.load('combo_labels_train-127.pt')
X_test = torch.load('X_test_combo-326.pt')
combo_labels_test = torch.load('combo_labels_test-326.pt')

print(X_test.shape)
train_dataset = TensorDataset(X_train, combo_labels_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test, combo_labels_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(CNN, self).__init__()

        # CNN layers
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # flatten the CNN's output and pass it directly to FC layers
        self.fc1 = nn.Linear(256*7, 128)

        self.fc_combo = nn.Linear(128, 48)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)

        # Flatten the output from the CNN
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        combo_out = self.fc_combo(x)

        return combo_out



best_accuracy = 0.0
patience_counter = 0
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []
test_loss_list = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    correct_test = 0
    total_train = 0
    total_test_loss = 0
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
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}")
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
    print(f"   Test Loss: {test_loss}", f"  Test Accuracy: {test_accuracy}%")
    #Early stop
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
#Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.title('Loss vs. epochs', fontsize=22)
plt.xlabel('Epochs', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.legend(prop={'size': 22})
plt.tick_params(axis='both', labelsize=22)
# Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy_list, label='Train Accuracy')
plt.plot(test_accuracy_list, label='Test Accuracy')
plt.title('Accuracy vs. epochs', fontsize=22)
plt.xlabel('Epochs', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.legend(prop={'size': 22})
plt.tick_params(axis='both', labelsize=22)
plt.tight_layout()
plt.show()