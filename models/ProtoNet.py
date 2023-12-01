#source environment room 326: 16 experiments
#target environment room 127: 9 experiments
#target environment room 127 with different person : 6 experiments

#set num_experiments in 'divide_into_target_experiments' function with corresponding number when doing different tasks
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#############Train datasets############
X_loaded = torch.load('X_tensor_final326.pt')  # default is using 5 sensor
#X_loaded = torch.load('X_tensor_1sensor_326.pt')  #change the number of sensors like 2sensor/3sensro/4sensor
combo_labels_loaded = torch.load('combined_labels_tensor_final326.pt')
#combo_labels_loaded = torch.load('combined_labels_tensor_1sensor_326.pt')

#############Test datasets(different room/person)############
X_test_full_loaded = torch.load('X_tensor_final127.pt')
#X_test_full_loaded = torch.load('X_tensor_1sensor_127.pt')
test_combo_labels_loaded = torch.load('test_combined_labels_tensor_final127.pt')
#test_combo_labels_loaded = torch.load('test_combined_labels_tensor_1sensor_127.pt')

#data of new person
X_test_full_person = torch.load('X_tensor_person.pt')
test_combo_labels_person = torch.load('test_combined_labels_tensor_person.pt')

# print(len(X_loaded))   #4 * 12 * 16 = 768
# print(len(action_labels_loaded)) #768


#source environment
def divide_into_source_experiments(data, labels, num_experiments=16):
    experiment_size = len(data) // num_experiments
    experiments_data = [data[i:i + experiment_size] for i in range(0, len(data), experiment_size)]
    experiments_labels = [labels[i:i + experiment_size] for i in range(0, len(labels), experiment_size)]

    return experiments_data, experiments_labels
#target environment
def divide_into_target_experiments(data, labels, num_experiments=9):
    experiment_size = len(data) // num_experiments
    experiments_data = [data[i:i + experiment_size] for i in range(0, len(data), experiment_size)]
    experiments_labels = [labels[i:i + experiment_size] for i in range(0, len(labels), experiment_size)]

    return experiments_data, experiments_labels
#Input for traning and testing
X_source, combo_labels = divide_into_source_experiments(X_loaded, combo_labels_loaded)
X_target, test_combo_labels = divide_into_target_experiments(X_test_full_loaded, test_combo_labels_loaded)
#Test for different person
#X_target, test_combo_labels = divide_into_target_experiments(X_test_full_person, test_combo_labels_person)

# Combine X_loaded and combo_labels_loaded into a single dataset
dataset = list(zip(X_source, combo_labels))

# Split the dataset into 80% training and 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Separate the datasets back into inputs and labels
X_train, combo_labels_train = zip(*train_dataset)
X_val, combo_labels_val = zip(*val_dataset)

# Convert lists back to tensors
X_train = torch.stack(X_train)
combo_labels_train = torch.stack(combo_labels_train)

X_val = torch.stack(X_val)
combo_labels_val = torch.stack(combo_labels_val)

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
        #out = x.unsqueeze(1)
        out = x
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layers(out)
        out = out.mean(2)
        out = self.fc(out)
        return out

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMFeatureExtractor, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(125, hidden_dim, batch_first=True,num_layers=1,bidirectional=False)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        ########################BiLSTM ########################
        # self.lstm = nn.LSTM(125, hidden_dim, batch_first=True, num_layers=1, bidirectional=True)
        # # Fully connected layer
        # self.fc = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)

        # Use the hidden state from the last time step as the feature
        feature_vector = hn[-1]

        ###########BiLSTM ###########################
        #feature_vector = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)


        # Fully connected layer
        output = self.fc(feature_vector)

        return output


class ProtoNet(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(ProtoNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2))
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # # the output size after convolutions is [batch_size, 256, length].
        # # use the LSTM to process the length dimension.
        # hidden_size = 256
        ## if use BiLSTM, bidirectional=True
        # self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        #
        #
        # # Since the LSTM is now bidirectional, the output size will be 2 * hidden_size
        # self.fc = nn.Linear( hidden_size, 48)
        # ################# Using CNN ##########
        self.fc1 = nn.Linear(256*7, 128)

        self.fc_combo = nn.Linear(128, 48)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        ######################## CNN-LSTM ######################
        # x = x.transpose(1, 2)  # this swaps the 256 and 7 dimensions
        # # LSTM expects input of shape [batch_size, seq_len, input_size]
        # # Get only the output of the last sequence
        # lstm_out, _ = self.lstm(x)
        #
        # x = lstm_out[:, -1, :]
        #
        # x = self.fc(x)
        ################using CNN  ###############
        #Flatten the output from the CNN
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc_combo(x)
        return x

def euclidean_dist(x, y):
  #Computes euclidean distance between x and y
  n = x.size(0)  #get the 0. dimension of x, which is number of samples of selected support data and query data
  m = y.size(0)  #get the 0. dimension of y, which is number of selected classes
  d = x.size(1)  #get the number of features(48)
  assert d == y.size(1)
  #print("input size",x.shape)
  #make x and y the same shape(n,m,d)
  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)
  # sum of (x-y)*(x-y), care the distance, but not the exact euclidean distance, so don't use Square root
  return torch.pow(x - y, 2).sum(2)

def prototypical_loss(input, target, n_support): #input uses embeddings, target uses labels.
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    unique_values = torch.unique(target_cpu) #get different classes from repeat classes(depends on n_support)
    #print("Unique values in target_cpu:", unique_values)

    # Create a mapping from original class labels to continuous labels
    # Make sure that even the real labels are not continous, they can still be used
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_values)}#create a dictionary, key is the unique values
    for original_label, new_label in label_mapping.items():
        target_cpu[target_cpu == original_label] = new_label
    def compute_prototypes(input, target, n_support):
        classes = torch.unique(target)
        n_classes = len(classes)  #number of unique selected classes
        prototypes = torch.zeros((n_classes, input.size(1)))   #initialize for selected  classes/prototypes, each has 48 feature dimensions.
        #print("number of prototypes",prototypes)
        for i, label in enumerate(classes):
            indices = torch.nonzero(target == label).squeeze()   #make sure tensor is 1D.
            prototypes[i] = input[indices[:n_support]].mean(0)   #using features of n_support data(selected classes) / all classes(here is 48)
        return prototypes

    prototypes = compute_prototypes(input_cpu, target_cpu, n_support) #[selected classes][all classes]

    dists = euclidean_dist(input_cpu, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)

    loss_val = -log_p_y.gather(1, target_cpu.view(-1, 1)).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(1)
    #print("y hat",y_hat)
    acc_val = torch.eq(y_hat, target_cpu).float().mean()


    return loss_val,  acc_val ,y_hat, target_cpu


def get_samples_for_episode(X_list, labels_list, n_support, max_n_way, random_n_way=True):
    # print('numbel of whole experiments',len(X_list))
    # according to experiments list to chose n support experiments
    chosen_experiment_indices = random.sample(range(len(X_list)), n_support)

    support_x = [X_list[idx] for idx in chosen_experiment_indices]
    support_y = [labels_list[idx] for idx in chosen_experiment_indices]
    # print("number of n_support experiments",len(support_x))
    # print("number of n_support labels in experiments",len(support_y))
    # Rest of experiments would be query sets
    query_experiment_indices = [idx for idx in range(len(X_list)) if idx not in chosen_experiment_indices]
    query_x = [X_list[idx] for idx in query_experiment_indices]
    query_y = [labels_list[idx] for idx in query_experiment_indices]

    # Combine all samples of data and labels in support and then select class randomly
    combined_support_x = torch.cat(support_x, dim=0)
    combined_support_y = torch.cat(support_y, dim=0)
    #print("number of all samples from support set",len(combined_support_x))
    unique_labels = torch.unique(combined_support_y).tolist()
    #print("unique labels",unique_labels)
    ##################### random select labels###################
    if random_n_way:
        n_way = random.randint(2, min(max_n_way, len(unique_labels)))

    else:
        n_way = max_n_way

    selected_labels = random.sample(unique_labels, n_way)
    ######################  choose labels with fixed number             ###################
    # if random_n_way:
    #     #n_way = len(unique_labels[::2])
    #
    #     selected_labels = unique_labels[::4]
    # else:
    #     n_way = max_n_way
    #     selected_labels = random.sample(unique_labels, n_way)

    #print("selected labels",selected_labels)

    # According to selected labels to choose corresponding samples
    final_support_x = torch.cat([combined_support_x[combined_support_y == label] for label in selected_labels], dim=0)
    final_support_y = torch.cat([combined_support_y[combined_support_y == label] for label in selected_labels], dim=0)

    combined_query_x = torch.cat(query_x, dim=0)
    combined_query_y = torch.cat(query_y, dim=0)

    final_query_x = torch.cat([combined_query_x[combined_query_y == label] for label in selected_labels], dim=0)
    final_query_y = torch.cat([combined_query_y[combined_query_y == label] for label in selected_labels], dim=0)

    return final_support_x, final_support_y, final_query_x, final_query_y


# Hyperparameters
n_epochs = 200
lr = 0.001
n_way = 48
n_support = 5  #n shot (samples per class)
n_support_val = 3
n_support_test = 5

#paratmeter for single LSTM
input_dim = 125  # Dimension of each input data point
hidden_dim = 256# Hidden state dimension
output_dim = 48  # Output dimension


# Initialize the model, optimizer, and criterion
#CNN as featureextractor
model = ProtoNet()

#ResNet as featureextractor
#model = ResNet(input_size=125, output_size=48)

#Single LSTM as featureextractor
#model = LSTMFeatureExtractor(input_dim, hidden_dim, output_dim)

optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-3)

#Learning rate adaption, default target is loss of validation
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

all_precisions = []
all_recalls = []
all_f1s = []
all_losses = []
all_accs = []
all_average_train_acc =[]
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


# Training Loop
for epoch in range(n_epochs):
    model.train()

    support_x, support_y, query_x, query_y = get_samples_for_episode(X_train, combo_labels_train, n_support,
                                                                     max_n_way=n_way, random_n_way=True)

    # print('number of samples of support data:',len(support_x))
    # print('number of samples of query data:',len(query_x))
    # print("all support labels",support_y)
    # print("all query labels",query_y)
    support_x = support_x.squeeze(2).squeeze(2).to(device)
    query_x = query_x.squeeze(2).squeeze(2).to(device)
    support_y = support_y.to(device)
    query_y = query_y.to(device)

    # Zero gradients
    optimizer.zero_grad()

    # Get embeddings/features for both support and query set
    embeddings_support = model(support_x)
    embeddings_query = model(query_x)


    # Concatenate embeddings = [sum number of samples of support and query ][48]
    embeddings = torch.cat([embeddings_support, embeddings_query], 0)

    # Concatenate targets, all labels for training
    targets = torch.cat([support_y, query_y], 0)

    # Calculate loss
    loss, acc ,y_hat, target_cpu = prototypical_loss(embeddings, targets, n_support)

    # Backward
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    train_accuracies.append(acc.item())

    print("Epoch {}: Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, loss.item(), acc.item()))
    model.eval()
    with torch.no_grad():
        # Create a new episode for validation
        support_x_val, support_y_val, query_x_val, query_y_val = get_samples_for_episode(X_val, combo_labels_val,
                                                                                 n_support_val,
                                                                                max_n_way=n_way,random_n_way=False)

        # Add an extra dimension for the channel and send data to GPU if available
        support_x_val = support_x_val.squeeze(2).to(device)
        query_x_val = query_x_val.squeeze(2).to(device)
        support_y_val = support_y_val.to(device)
        query_y_val = query_y_val.to(device)

        # Get embeddings for both support and query set for validation
        embeddings_support_val = model(support_x_val)
        embeddings_query_val = model(query_x_val)

        # Concatenate embeddings
        embeddings_val = torch.cat([embeddings_support_val, embeddings_query_val], 0)

        # Concatenate targets
        targets_val = torch.cat([support_y_val, query_y_val], 0)

        # Calculate validation loss
        val_loss, val_acc ,y_hat, target_cpu= prototypical_loss(embeddings_val, targets_val, n_support_val)
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc.item())

    print(
        f"Epoch {epoch + 1}: Train Loss: {loss.item():.4f}, Train Acc: {acc.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}")

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

#Record the train accuracy
average_train_accuracy = sum(train_accuracies) / len(train_accuracies)
print(f"average train ACC: {average_train_accuracy:.4f}")
all_average_train_acc.append(average_train_accuracy)

#test the ability for transfor 10 times
for i in range(10):
    #Evaluation
    model.eval()


    support_x, support_y, query_x, query_y = get_samples_for_episode(X_target, test_combo_labels,
                                                             n_support_test, max_n_way= n_way,random_n_way=False)
    # print('number of samples of support data:',len(support_x))
    # print('number of samples of query data:',len(query_x))
    support_x = support_x.squeeze(2).squeeze(2).to(device)
    query_x = query_x.squeeze(2).squeeze(2).to(device)
    support_y = support_y.to(device)
    query_y = query_y.to(device)


    embeddings_support = model(support_x)
    embeddings_query = model(query_x)


    embeddings = torch.cat([embeddings_support, embeddings_query], 0)


    targets = torch.cat([support_y, query_y], 0)


    loss, acc ,y_hat, target_cpu= prototypical_loss(embeddings, targets, n_support_test)


    precision = precision_score(target_cpu, y_hat, average='macro')
    recall = recall_score(target_cpu, y_hat, average='macro')
    f1 = f1_score(target_cpu, y_hat, average='macro')
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)
    all_losses.append(loss.item())
    all_accs.append(acc.item())
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    print("Transfer Test: Loss: {:.4f}, Acc: {:.4f}".format(loss.item(), acc.item()))

    # ###########################Confusion metrics#############################
    # def decimal_formatter(x):
    #     if x == 0:
    #         return "0"
    #     return "{:.2f}".format(x)
    #
    #
    # y_true = target_cpu.numpy()
    # y_pred = y_hat.numpy()
    # cm = confusion_matrix(y_true, y_pred)
    # row_sums = cm.sum(axis=1)
    # cm_percentage = cm / row_sums[:, np.newaxis]
    # vfunc = np.vectorize(decimal_formatter)
    # cm_percentage_str = vfunc(cm_percentage)
    # #show the figure
    # plt.figure(figsize=(12, 12))
    # sns.heatmap(cm_percentage, annot=cm_percentage_str, cmap="Blues", cbar=False, fmt='s',
    #             annot_kws={"size": 10})
    # plt.xlabel('Predicted Labels', fontsize=16)
    # plt.ylabel('True Labels', fontsize=16)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=8)
    # plt.show()

for i in range(10):
    print(f"Iteration {i+1}:")
    print(f"Precision: {all_precisions[i]:.4f}, Recall: {all_recalls[i]:.4f}, F1 Score: {all_f1s[i]:.4f}")
    print(f"Transfer Test: Loss: {all_losses[i]:.4f}, Acc: {all_accs[i]:.4f}")
    print("-" * 50)  # just a separator for better visualization
avg_precision = sum(all_precisions) / len(all_precisions)
avg_recall = sum(all_recalls) / len(all_recalls)
avg_f1 = sum(all_f1s) / len(all_f1s)
avg_loss = sum(all_losses) / len(all_losses)
avg_acc = sum(all_accs) / len(all_accs)
average_train_acc = sum(all_average_train_acc) / len(all_average_train_acc)

print(f"average train ACC: {average_train_acc:.4f}")
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1 Score:", avg_f1)
print("Average Loss:", avg_loss)
print("Average Accuracy:", avg_acc)
print("-" * 50)

max_precision = max(all_precisions)
max_recall = max(all_recalls)
max_f1 = max(all_f1s)
max_loss = max(all_losses)
max_acc = max(all_accs)

print("Maximum Precision:", max_precision)
print("Maximum Recall:", max_recall)
print("Maximum F1 Score:", max_f1)
print("Maximum Loss:", max_loss)
print("Maximum Accuracy:", max_acc)
print("-" * 50)

min_precision = min(all_precisions)
min_recall = min(all_recalls)
min_f1 = min(all_f1s)
min_loss = min(all_losses)
min_acc = min(all_accs)

print("Minimum Precision:", min_precision)
print("Minimum Recall:", min_recall)
print("Minimum F1 Score:", min_f1)
print("Minimum Loss:", min_loss)
print("Minimum Accuracy:", min_acc)
