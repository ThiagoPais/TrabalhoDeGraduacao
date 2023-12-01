# single LSTM
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
print(X_test.shape)
print(combo_labels_test.shape)
X_train = X_train.unsqueeze(-1)
X_test = X_test.unsqueeze(-1)


train_dataset = TensorDataset(X_train, combo_labels_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test, combo_labels_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True,num_layers=1,bidirectional=False)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)

        # Use the hidden state from the last time step as the feature
        feature_vector = hn[-1]
        #feature_vector = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)

        # Fully connected layer
        output = self.fc(feature_vector)

        return output

best_accuracy = 0.0
patience_counter = 0
input_dim = 1  # Dimension of each input data point
hidden_dim = 256# Hidden state dimension
output_dim = 48  # Output dimension
train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []
test_loss_list = []
model = LSTM(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

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
plt.title('Loss vs. epochs', fontsize=22)
plt.xlabel('Epochs', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.legend(prop={'size': 22})
plt.tick_params(axis='both', labelsize=22)

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
