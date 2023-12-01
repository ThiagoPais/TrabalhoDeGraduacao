###for evaluation
import matplotlib.pyplot as plt
import numpy as np


models = ['CNN', 'LSTM', 'BiLSTM', 'CNN-LSTM', "CNN-BiLSTM", "ResNet" ]
accuracy = [84.60, 81.43, 81.06, 77.59, 74.12, 65.50]
f1_scores = [0.8433, 0.8103, 0.8056, 0.7713,0.7347,0.6504]

plt.rcParams.update({'font.size': 20})
fig, ax1 = plt.subplots(figsize=(10, 6))


barWidth = 0.3
r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]


ax1.bar(r1, accuracy, color='blue', width=barWidth, edgecolor='grey', label='Accuracy (%)')
ax1.set_xlabel('Feature Extractors', fontsize=18)

ax1.set_ylabel('Accuracy (%)', fontsize=18)
ax1.set_ylim([0, 100])


for i in range(len(r1)):
    ax1.text(r1[i], accuracy[i] + 0.05, f"{accuracy[i]:.2f}%", ha='center', va='bottom', fontsize=18)


ax2 = ax1.twinx()
ax2.bar(r2, f1_scores, color='red', width=barWidth, edgecolor='grey', label='F1 Score')
ax2.set_ylabel('F1 Score', fontsize=18)
ax2.set_ylim([0, 1])


for i in range(len(r2)):
    ax2.text(r2[i], f1_scores[i] + 0.04, f"{f1_scores[i]:.3f}", ha='center', va='bottom', fontsize=18)

plt.title('Performance Metrics for Different Models', fontsize=20)

plt.xticks([r + barWidth/2 for r in range(len(accuracy))], models)

plt.tight_layout()
plt.show()



models = ['1', '2', '3', '4', "5" ]
accuracy = [59.39, 74.88, 81.80, 82.84, 84.60]
f1_scores = [0.5883, 0.7493, 0.8169, 0.8273,0.843]

plt.rcParams.update({'font.size': 20})

fig, ax1 = plt.subplots(figsize=(10, 6))

barWidth = 0.3
r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]

ax1.bar(r1, accuracy, color='blue', width=barWidth, edgecolor='grey', label='Accuracy (%)')
ax1.set_xlabel('Number of Sensors', fontsize=18)

ax1.set_ylabel('Accuracy (%)', fontsize=18)
ax1.set_ylim([0, 100])

for i in range(len(r1)):
    ax1.text(r1[i], accuracy[i] + 0.05, f"{accuracy[i]:.2f}%", ha='center', va='bottom', fontsize=18)


ax2 = ax1.twinx()
ax2.bar(r2, f1_scores, color='red', width=barWidth, edgecolor='grey', label='F1 Score')
ax2.set_ylabel('F1 Score', fontsize=18)
ax2.set_ylim([0, 1])

for i in range(len(r2)):
    ax2.text(r2[i], f1_scores[i] + 0.04, f"{f1_scores[i]:.3f}", ha='center', va='bottom', fontsize=18)

plt.title('Performance Metrics for Different Number of Sensors', fontsize=20)

plt.xticks([r + barWidth/2 for r in range(len(accuracy))], models)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

models = ['1', '2', '3', '4', "5"]
accuracy = [59.39, 74.88, 81.80, 82.84, 84.60]
cost = [12.6, 25.2, 37.8, 50.4, 63]

# Compute combined scores for each model across a range of weight coefficients
weights = np.linspace(0, 1, 11)  # Weight coefficients from 0 to 1 in intervals of 0.1
combined_scores = []

for model_accuracy, model_f1 in zip(accuracy, cost):
    model_scores = []
    for w in weights:
        model_score = w * model_accuracy - (1 - w) * model_f1   # Convert F1 score to percentage for calculation
        model_scores.append(model_score)
    combined_scores.append(model_scores)

# Plot
plt.figure(figsize=(12, 8))

for model, model_scores in zip(models, combined_scores):
    plt.plot(weights, model_scores, label=f"Number of Sensors {model}", marker='o')

plt.title("Combined Performance Metrics Across Weight Coefficients")
plt.xlabel("Weight Coefficient for Accuracy and Cost")
plt.ylabel("Combined Score ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

