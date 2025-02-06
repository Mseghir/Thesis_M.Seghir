import numpy as np
import matplotlib.pyplot as plt

# Define moving average function
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Scores for each model
dt_random_split_scores = [0.66, 0.68, 0.67, 0.67, 0.66]
dt_temporal_split_scores = [0.61, 0.67, 0.69, 0.67, 0.69]

xgboost_random_split_scores = [0.63, 0.63, 0.61, 0.61, 0.64]
xgboost_temporal_split_scores = [0.58, 0.61, 0.65, 0.65, 0.64]

random_forest_random_split_scores = [0.62, 0.62, 0.61, 0.61, 0.62]
random_forest_temporal_split_scores = [0.59, 0.58, 0.64, 0.63, 0.62]

# Fold indices
folds = [1, 2, 3, 4, 5]

# Plotting Decision Tree with Moving Average
plt.figure(figsize=(10, 6))
models = ["DT, random split", "DT, temporal split"]
scores = [dt_random_split_scores, dt_temporal_split_scores]
colors = ['blue', 'orange']

for model, model_scores, color in zip(models, scores, colors):
    # Plot the original F1 scores
    plt.plot(folds, model_scores, marker='o', label=model, color=color)
    # Calculate and plot the moving average with the same color
    moving_avg = moving_average(model_scores)
    plt.plot(folds[1:len(moving_avg)+1], moving_avg, linestyle='--', color=color, label=f'{model} MA')

# Add labels and title
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("F1 Scores per Fold for the DT")
plt.xticks(folds)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# Plotting XGBoost with Moving Average
plt.figure(figsize=(10, 6))
models = ["XGBoost, random split", "XGBoost, temporal split"]
scores = [xgboost_random_split_scores, xgboost_temporal_split_scores]

for model, model_scores, color in zip(models, scores, colors):
    # Plot the original F1 scores
    plt.plot(folds, model_scores, marker='o', label=model, color=color)
    # Calculate and plot the moving average with the same color
    moving_avg = moving_average(model_scores)
    plt.plot(folds[1:len(moving_avg)+1], moving_avg, linestyle='--', color=color, label=f'{model} MA')

# Add labels and title
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("F1 Scores per Fold for XGBoost")
plt.xticks(folds)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# Plotting Random Forest with Moving Average
plt.figure(figsize=(10, 6))
models = ["Random Forest, random split", "Random Forest, temporal split"]
scores = [random_forest_random_split_scores, random_forest_temporal_split_scores]

for model, model_scores, color in zip(models, scores, colors):
    # Plot the original F1 scores
    plt.plot(folds, model_scores, marker='o', label=model, color=color)
    # Calculate and plot the moving average with the same color
    moving_avg = moving_average(model_scores)
    plt.plot(folds[1:len(moving_avg)+1], moving_avg, linestyle='--', color=color, label=f'{model} MA')

# Add labels and title
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("F1 Scores per Fold for Random Forest")
plt.xticks(folds)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
