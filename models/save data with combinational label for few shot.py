import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from amp_time_functionzidian_success import amp_time
import random
from sklearn.model_selection import train_test_split
import numpy as np
# from amp_time_functionzidian_success import amp_time
data_dict = {
    "sit": {},
    "stand": {},
    "walk": {},
    "empty": {}
}

actions = ["sit", "stand", "walk", "empty"]
directions = ["0 o'clock", "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock",
              "5 o'clock", "6 o'clock", "7 o'clock", "8 o'clock", "9 o'clock",
              "10 o'clock", "11 o'clock"]
###########source-train(support set), target-test(query set)
base_path_source = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/0928itiv 326/insert/i'
base_path_target = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/0929itiv 127/insert/i'

data_dict_train = {}
data_dict_test  = {}
def get_combined_label(action, direction):
    return action_label_mapping[action] * 12 + direction_label_mapping[direction]

######## create dictionary for train##########
for ex in range(1, 17):
    experiment_key = f"experiment_{ex}"
    data_dict_train[experiment_key] = {}

    for action in actions:
        data_dict_train[experiment_key][action] = {}

        file_in = f'{action}{ex}.csv'
        file_path = base_path_source + file_in
        direction_data = amp_time(file_path, visualize=False)
        ########################################################################### take data out of direction_data and store them in data_dict_train
        #############         structure：data_dict_tain[experiment][action][direction] = data
        #print(f"Output of amp_time for file {file_in}:")
        #print(direction_data)
        if 'direction_amplitudes' in direction_data:
            for dir_name, dir_values in direction_data['direction_amplitudes'].items():
                #print(f"Data length for {file_in} - {dir_name}: {len(dir_values)}")  # Print the length of the data
                data_dict_train[experiment_key][action][dir_name] = dir_values
        else:
            print(f"Warning: 'direction_amplitudes' key not found for file {file_in}. Skipping this file.")

######## create dictionary for test##########
for ex in range(1, 10):
    experiment_key = f"experiment_{ex}"
    data_dict_test[experiment_key] = {}

    for action in actions:
        data_dict_test[experiment_key][action] = {}

        file_in = f'{action}{ex}.csv'
        file_path = base_path_target + file_in
        direction_data = amp_time(file_path, visualize=False)
        ########################################################################### take data out of direction_data and store them in data_dict_test
        #############         structure：data_dict_tain[experiment][action][direction] = data
        if 'direction_amplitudes' in direction_data:
            for dir_name, dir_values in direction_data['direction_amplitudes'].items():
                data_dict_test[experiment_key][action][dir_name] = dir_values
        else:
            print(f"Warning: 'direction_amplitudes' key not found for file {file_in}. Skipping this file.")

action_label_mapping = {
    "sit": 0,
    "stand": 1,
    "walk": 2,
    "empty": 3
}
direction_label_mapping = {
    "0 o'clock": 0,
    "1 o'clock": 1,
    "2 o'clock": 2,
    "3 o'clock": 3,
    "4 o'clock": 4,
    "5 o'clock": 5,
    "6 o'clock": 6,
    "7 o'clock": 7,
    "8 o'clock": 8,
    "9 o'clock": 9,
    "10 o'clock": 10,
    "11 o'clock": 11,

}

# create list for training samples(source environment)
train_data_list = []  # This will store the actual data
combined_labels_list = []
for experiment, actions_data in data_dict_train.items():
    for action, directions_data in actions_data.items():
        for direction, data in directions_data.items():
            train_data_list.append(data)
            # Append the combined label for this data
            combined_labels_list.append(get_combined_label(action, direction))


# Convert data and labels to PyTorch tensors
data_array = np.array(train_data_list) #Creating a tensor from a list of numpy.ndarrays is extremely slow， so change list into numpy array first
X = torch.tensor(data_array, dtype=torch.float32).unsqueeze(1)
combined_labels = torch.tensor(combined_labels_list, dtype=torch.long)

train_dataset = TensorDataset(X, combined_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# create list for test samples(target environment)
test_data_list = []  # This will store the actual test data
test_combined_labels_list = []
for experiment, actions_data in data_dict_test.items():
    for action, directions_data in actions_data.items():
        for direction, data in directions_data.items():
            test_data_list.append(data)
            # Append the combined label for this test data
            test_combined_labels_list.append(get_combined_label(action, direction))

# Convert test data and labels to PyTorch tensors
X_test_full = torch.tensor(np.array(test_data_list), dtype=torch.float32).unsqueeze(1)
test_combined_labels = torch.tensor(test_combined_labels_list, dtype=torch.long)

# # Ensure shapes are as expected
# assert X_test_full.shape[2] == 125, f"Unexpected X_test_full shape: {X_test_full.shape}"

# Create DataLoader for the full test dataset
test_full_dataset = TensorDataset(X_test_full, test_combined_labels)
test_full_loader = DataLoader(test_full_dataset, batch_size=32, shuffle=False)


torch.save(X, 'X_tensor_1sensor_326.pt')
torch.save(combined_labels, 'combined_labels_tensor_1sensor_326.pt')

torch.save(X_test_full, 'X_tensor_1sensor_127.pt')
torch.save(test_combined_labels, 'test_combined_labels_tensor_1sensor_127.pt')
