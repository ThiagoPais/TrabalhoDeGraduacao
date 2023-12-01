# add LSTM right after CNN
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from amp_time_functionzidian_success import amp_time
from sklearn.model_selection import train_test_split
import numpy as np

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

# def add_noise(x, noise_level=0.05):
#     """Add Gaussian noise to the time series"""
#     noise = np.random.randn(len(x))
#     return x + noise_level * noise
#
#
# def random_shift(x, shift_fraction=0.2):
#     """Randomly shift the time series"""
#     max_shift = int(shift_fraction * len(x))
#     shift = np.random.randint(-max_shift, max_shift)
#     return np.roll(x, shift)
#
#
# def random_stretch(x, stretch_fraction=0.2):
#     """Randomly stretch or compress the time series"""
#     stretch_factor = 1 + np.random.uniform(-stretch_fraction, stretch_fraction)
#     new_length = int(len(x) * stretch_factor)
#     # Create new x coordinates for interpolation
#     x_new = np.linspace(0, len(x), new_length)
#     return np.interp(x_new, np.arange(len(x)), x)
#
#
# def random_slice(x, slice_length=0.8):
#     """Randomly select a subsequence from the time series"""
#     start_idx = np.random.randint(0, int(len(x) * (1 - slice_length)))
#     end_idx = start_idx + int(len(x) * slice_length)
#     return x[start_idx:end_idx]


base_path_output = 'D:/KIT/毕业设计/Boyang-Master-thesis/data/0928itiv 326/insert/i'

data_dict = {}

for ex in range(1, 17):
    experiment_key = f"experiment_{ex}"
    data_dict[experiment_key] = {}

    for action in actions:
        data_dict[experiment_key][action] = {}

        file_in = f'{action}{ex}.csv'
        file_path = base_path_output + file_in
        direction_data = amp_time(file_path, visualize=False)
            # data agumentation
        # if 'direction_amplitudes' in direction_data:
        #         for dir_name, dir_values in direction_data['direction_amplitudes'].items():
        #             augmented_values = add_noise(dir_values)
        #     #         #augmented_values = random_shift(augmented_values)
        #     #         augmented_values = random_shift(dir_values)
        #     #         #augmented_values = random_stretch(dir_values)  #not useful
        #     #         #augmented_values = random_slice(dir_values)  #not useful
        #     #
        #     #         data_dict[experiment_key][action][dir_name] = augmented_values
        #         else:
        #             print(f"Warning: 'direction_amplitudes' key not found for file {file_in}. Skipping this file.")
        ###########################################################################
        if 'direction_amplitudes' in direction_data:
            for dir_name, dir_values in direction_data['direction_amplitudes'].items():
                data_dict[experiment_key][action][dir_name] = dir_values
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
combined_labels_mapping = {}
label_counter = 0
for action in actions:
    for direction in directions:
        combined_label = f"{action}-{direction}"
        combined_labels_mapping[combined_label] = label_counter
        label_counter += 1
# Lists to store data and labels
data_list = []  # This will store the actual data
combined_labels_list = []  # This will store the combined labels

for experiment, actions_data in data_dict.items():
    for action, directions_data in actions_data.items():
        for direction, data in directions_data.items():
            data_list.append(data)

            # Append the combined label for this data
            combined_label = f"{action}-{direction}"
            combined_labels_list.append(combined_labels_mapping[combined_label])


# Convert data and labels to PyTorch tensors
X = torch.tensor(np.array(data_list), dtype=torch.float32)
combined_labels = torch.tensor(combined_labels_list, dtype=torch.long)


# Split data into training and testing sets
X_train, X_test, combined_labels_train, combined_labels_test = train_test_split(
    X, combined_labels, test_size=0.2, stratify=combined_labels)

train_dataset = TensorDataset(X_train, combined_labels_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test, combined_labels_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

torch.save(X_train, 'X_train_combo-326.pt')
torch.save(X_test, 'X_test_combo-326.pt')
torch.save(combined_labels_train, 'combo_labels_train-326.pt')
torch.save(combined_labels_test, 'combo_labels_test-326.pt')
