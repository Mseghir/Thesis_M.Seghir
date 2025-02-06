# Importing libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree
import optuna #doing baysian optimalization instead of gridsearch 

# Custom function to create stratified time-series splits
def stratified_time_series_split(X, y, n_splits=5):
    # Create a list of indices to hold the splits
    indices = []

    # Initialize the StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)

    # Generate the indices for stratified time-series split
    for train_index, val_index in stratified_kfold.split(X, y):
        # Ensure the maintainance of the temporal order
        # slice the indices based on time (first train, then test)
        indices.append((train_index, val_index))
        
    return indices

# Load the balanced temporal training, validation and test datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')
x_test_temp = pd.read_csv('x_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv')

# Load the balanced random training, validation and test datasets
X_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv')
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv')
x_test_rand = pd.read_csv('x_test_rand.csv')
y_test_rand = pd.read_csv('y_test_rand.csv')

# Initialize the decision tree classifier with random state 777
def objective(trial):
    # Define the hyperparameters to tune
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    # Create the model with suggested hyperparameters
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=777
    )

    # Perform cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=777)
    f1_scores = cross_val_score(model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')

    # Print the F1 scores for each fold
    print(f"F1 scores across folds (Random Set): {f1_scores}")
    print(f"Mean F1 score (Random Set): {np.mean(f1_scores)}")

    # Return the mean F1 score as the objective to minimize
    return np.mean(f1_scores)

# Optimize hyperparameters for the random set
study_rand = optuna.create_study(direction='maximize', study_name='Random Set Optimization')
study_rand.optimize(objective, n_trials=50, n_jobs=-1)

# Print the best parameters and score for the random set
print("Best Parameters (Random Set):", study_rand.best_params)
print("Best F1 Score (Random Set):", study_rand.best_value)

# Define a separate objective for the temporal set
def objective_temp(trial):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=777
    )

    stratified_splits = stratified_time_series_split(X_temp_balanced, y_temp_balanced, n_splits=5)
    f1_scores = []
    for train_index, val_index in stratified_splits:
        X_train, X_val = X_temp_balanced.iloc[train_index], X_temp_balanced.iloc[val_index]
        y_train, y_val = y_temp_balanced.iloc[train_index], y_temp_balanced.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred))
    
    # Print the F1 scores for each fold
    print(f"F1 scores across folds (Temporal Set): {f1_scores}")
    print(f"Mean F1 score (Temporal Set): {np.mean(f1_scores)}")

    return np.mean(f1_scores)

# Optimize hyperparameters for the temporal set
study_temp = optuna.create_study(direction='maximize', study_name='Temporal Set Optimization')
study_temp.optimize(objective_temp, n_trials=50, n_jobs=-1)

# Print the best parameters and score for the temporal set
print("Best Parameters (Temporal Set):", study_temp.best_params)
print("Best F1 Score (Temporal Set):", study_temp.best_value)

# Random Validation Set 
best_model_rand = DecisionTreeClassifier(**study_rand.best_params, random_state=777)
best_model_rand.fit(X_rand_balanced, y_rand_balanced)
val_rand_predictions = best_model_rand.predict(x_val_rand)

# Calculate evaluation metrics on the validation set 
accuracy_rand = accuracy_score(y_val_rand, val_rand_predictions)
precision_rand = precision_score(y_val_rand, val_rand_predictions)
recall_rand = recall_score(y_val_rand, val_rand_predictions)
f1_rand = f1_score(y_val_rand, val_rand_predictions)
roc_auc_rand = roc_auc_score(y_val_rand, best_model_rand.predict_proba(x_val_rand)[:, 1])
conf_matrix_rand = confusion_matrix(y_val_rand, val_rand_predictions)

# Print evaluation metrics
print("Validation Metrics (Random Set):")
print(f"Accuracy: {accuracy_rand}")
print(f"Precision: {precision_rand}")
print(f"Recall: {recall_rand}")
print(f"F1-Score: {f1_rand}")
print(f"AUC-ROC: {roc_auc_rand}")
print(f"Confusion Matrix:\n{conf_matrix_rand}")

# Temporal Validation Set
best_model_temp = DecisionTreeClassifier(**study_temp.best_params, random_state=777)
best_model_temp.fit(X_temp_balanced, y_temp_balanced)
val_temp_predictions = best_model_temp.predict(x_val_temp)

# Calculate evaluation metrics on the temporal validation set
accuracy_temp = accuracy_score(y_val_temp, val_temp_predictions)
precision_temp = precision_score(y_val_temp, val_temp_predictions)
recall_temp = recall_score(y_val_temp, val_temp_predictions)
f1_temp = f1_score(y_val_temp, val_temp_predictions)
roc_auc_temp = roc_auc_score(y_val_temp, best_model_temp.predict_proba(x_val_temp)[:, 1])
conf_matrix_temp = confusion_matrix(y_val_temp, val_temp_predictions)

# Print evaluation metrics
print("Validation Metrics (Temporal Set):")
print(f"Accuracy: {accuracy_temp}")
print(f"Precision: {precision_temp}")
print(f"Recall: {recall_temp}")
print(f"F1-Score: {f1_temp}")
print(f"AUC-ROC: {roc_auc_temp}")
print(f"Confusion Matrix:\n{conf_matrix_temp}")

# Combine the training and validation datasets for both random and temporal splits
X_combined_rand = pd.concat([X_rand_balanced, x_val_rand], axis=0)
y_combined_rand = pd.concat([y_rand_balanced, y_val_rand], axis=0)

X_combined_temp = pd.concat([X_temp_balanced, x_val_temp], axis=0)
y_combined_temp = pd.concat([y_temp_balanced, y_val_temp], axis=0)

# final model fit on the best models on the combined training and validation sets
final_model_rand = DecisionTreeClassifier(**study_rand.best_params, random_state=777)
final_model_rand.fit(X_combined_rand, y_combined_rand)

final_model_temp = DecisionTreeClassifier(**study_temp.best_params, random_state=777)
final_model_temp.fit(X_combined_temp, y_combined_temp)

# Permutation Importance for Random Set
perm_importance_rand = permutation_importance(final_model_rand, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
rand_importance_df = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Importance': perm_importance_rand.importances_mean
})

# I dont want a cluttered plot... So using a cutoff of 0.001 and -0.001
rand_importance_df = rand_importance_df[(rand_importance_df['Importance'] > 0.001) | (rand_importance_df['Importance'] < -0.001)]
rand_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plotting Permutation Importance for Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rand_importance_df)
plt.title('Permutation Importance DT (Random Set)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Permutation Importance for Temporal Set
perm_importance_temp = permutation_importance(final_model_temp, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
temp_importance_df = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp.importances_mean
})

# I dont want a cluttered plot... So using a cutoff of 0.001 and -0.001
temp_importance_df = temp_importance_df[(temp_importance_df['Importance'] > 0.001) | (temp_importance_df['Importance'] < -0.001)]
temp_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plotting Permutation Importance for Temporal Set 
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=temp_importance_df)
plt.title('Permutation Importance DT (Temporal Set)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Visualization of Optimization History
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_rand)
plt.title('Optimization History DT (Random Set)')
plt.show()

plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_temp)
plt.title('Optimization History DT (Temporal Set)')
plt.show()

# FINAL EVALUATION ON THE TEST SET

# Test the best model from the Random Split on the Random Test Set
test_rand_predictions = final_model_rand.predict(x_test_rand)

# Calculate the evaluation metrics from the random split on the Random Test Set
accuracy_test_rand = accuracy_score(y_test_rand, test_rand_predictions)
precision_test_rand = precision_score(y_test_rand, test_rand_predictions)
recall_test_rand = recall_score(y_test_rand, test_rand_predictions)
f1_test_rand = f1_score(y_test_rand, test_rand_predictions)
roc_auc_test_rand = roc_auc_score(y_test_rand, final_model_rand.predict_proba(x_test_rand)[:, 1])
conf_matrix_test_rand = confusion_matrix(y_test_rand, test_rand_predictions)

# Print the test metrics for the Random Test Set
print("Test Metrics (Random Set):")
print(f"Accuracy: {accuracy_test_rand}")
print(f"Precision: {precision_test_rand}")
print(f"Recall: {recall_test_rand}")
print(f"F1-Score: {f1_test_rand}")
print(f"AUC-ROC: {roc_auc_test_rand}")
print(f"Confusion Matrix:\n{conf_matrix_test_rand}")

# Test the best model from the Temporal Split on the Temporal Test Set
test_temp_predictions = final_model_temp.predict(x_test_temp)

# Calculate the evaluation metrics from the temporal split on the Temporal Test Set
accuracy_test_temp = accuracy_score(y_test_temp, test_temp_predictions)
precision_test_temp = precision_score(y_test_temp, test_temp_predictions)
recall_test_temp = recall_score(y_test_temp, test_temp_predictions)
f1_test_temp = f1_score(y_test_temp, test_temp_predictions)
roc_auc_test_temp = roc_auc_score(y_test_temp, final_model_temp.predict_proba(x_test_temp)[:, 1])
conf_matrix_test_temp = confusion_matrix(y_test_temp, test_temp_predictions)

# Print the test metrics for the Temporal Test Set
print("Test Metrics (Temporal Set):")
print(f"Accuracy: {accuracy_test_temp}")
print(f"Precision: {precision_test_temp}")
print(f"Recall: {recall_test_temp}")
print(f"F1-Score: {f1_test_temp}")
print(f"AUC-ROC: {roc_auc_test_temp}")
print(f"Confusion Matrix:\n{conf_matrix_test_temp}")



"""RESULTS ON ENTIRE DATASET:
Random Set - 5-Fold Cross-Validation F1 Scores: [0.78583618 0.79333478 0.78772157 0.79343838 0.7962248 ]
Random Set - Mean F1 Score: 0.7913111400594282

Temporal Set - Time Series Cross-Validation F1 Scores: [0.83591423 0.83464333 0.82290722 0.36111111 0.        ]
Temporal Set - Mean F1 Score: 0.5709151804656348

Fitting 5 folds for each of 90 candidates, totalling 450 fits
Best Parameters (Random Set): {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best F1 Score (Random Set): 0.6328442113577373

Fitting 5 folds for each of 90 candidates, totalling 450 fits
Best Parameters (Temporal Set): {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best F1 Score (Temporal Set): 0.7882314350466181

Validation Metrics (Random Set):
Accuracy: 0.7482416879795396
Precision: 0.8124018838304553
Recall: 0.7834973504920515
F1-Score: 0.7976878612716762
AUC-ROC: 0.7422321982040812
Confusion Matrix:
[[1576  717]
 [ 858 3105]]

Validation Metrics (Temporal Set):
Accuracy: 0.6120524296675192
Precision: 0.5796265788028556
Recall: 0.7020285999334885
F1-Score: 0.6349827041660401
AUC-ROC: 0.6153170047564198
Confusion Matrix:
[[1718 1531]
 [ 896 2111]]

Test Metrics (Random Set):
Accuracy: 0.7521
Precision: 0.8159
Recall: 0.7794
F1-Score: 0.7972
AUC-ROC: 0.7478
Confusion Matrix:
[[1657  688]
 [ 863 3049]]

Test Metrics (Temporal Set):
Accuracy: 0.6014
Precision: 0.5736
Recall: 0.6986
F1-Score: 0.6300
AUC-ROC: 0.6041
Confusion Matrix:
[[1640 1578]
 [ 916 2123]]

 RESULTS ON ONLY DATA AFTER 2010:

Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.5872534142640364, 0.5988212180746562, 0.6174944403261675, 0.6045262754123514, 0.6151480199923107]
Temporal Set - Mean F1 Score: 0.6046486736139045

Random Set - 5-Fold Cross-Validation F1 Scores: [0.60099864 0.61457419 0.61364663 0.60907889 0.5916781 ]
Random Set - Mean F1 Score: 0.6059952893726763

Fitting 5 folds for each of 90 candidates, totalling 450 fits
Best Parameters (Random Set): {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2}
Best F1 Score (Random Set): 0.5587684743342667

Fitting 5 folds for each of 90 candidates, totalling 450 fits
Best Parameters (Temporal Set): {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 10}
Best F1 Score (Temporal Set): 0.6968102670699478

Validation Metrics (Random Set):
Accuracy: 0.6842734225621415
Precision: 0.6109848484848485
Recall: 0.8458311484006292
F1-Score: 0.7094787772157466
AUC-ROC: 0.7700640153616602
Confusion Matrix:
[[1250 1027]
 [ 294 1613]]

Validation Metrics (Temporal Set):
Accuracy: 0.6472275334608031
Precision: 0.5002736726874658
Recall: 0.6188219363574814
F1-Score: 0.553268765133172
AUC-ROC: 0.6951930587441121
Confusion Matrix:
[[1794  913]
 [ 563  914]]

Test Metrics (Random Set):
Accuracy: 0.6869
Precision: 0.6119
Recall: 0.8340
F1-Score: 0.7059
AUC-ROC: 0.7701
Confusion Matrix:
[[1302  997]
 [ 313 1572]]

Test Metrics (Temporal Set):
Accuracy: 0.7502390057361377
Precision: 0.6752411575562701
Recall: 0.5671843349088453
F1-Score: 0.6165137614678899
AUC-ROC: 0.8149745087797263
Confusion Matrix:
[[2299  404]
 [ 641  840]]
 
 After removing temporal features and adding in new features to mitigate bias:

 Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.564638783269962, 0.558922558922559, 0.5999109924343569, 0.5898024804777218, 0.5590887517797816]
Temporal Set - Mean F1 Score: 0.5744727133768762

Random Set - 5-Fold Cross-Validation F1 Scores: [0.57380457 0.56237006 0.55584416 0.57376212 0.56713323]
Random Set - Mean F1 Score: 0.5665828290407753

Fitting 5 folds for each of 90 candidates, totalling 450 fits
Best Parameters (Random Set): {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2}
Best F1 Score (Random Set): 0.5041966882434151

Fitting 5 folds for each of 90 candidates, totalling 450 fits
Best Parameters (Temporal Set): {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best F1 Score (Temporal Set): 0.6773447913072583

Validation Metrics (Random Set):
Accuracy: 0.654144305307096
Precision: 0.6161290322580645
Recall: 0.7198492462311558
F1-Score: 0.6639629200463499
AUC-ROC: 0.7068069490471655
Confusion Matrix:
[[1048  714]
 [ 446 1146]]

Validation Metrics (Temporal Set):
Accuracy: 0.6666666666666666
Precision: 0.5565058032987171
Recall: 0.6991557943207981
F1-Score: 0.6197278911564627
AUC-ROC: 0.7279248690248248
Confusion Matrix:
[[1325  726]
 [ 392  911]]

Test Metrics (Random Set):
Accuracy: 0.6410256410256411
Precision: 0.5854341736694678
Recall: 0.6925115970841617
F1-Score: 0.6344869459623558
AUC-ROC: 0.7018253621900037
Confusion Matrix:
[[1105  740]
 [ 464 1045]]
 
Test Metrics (Temporal Set):
Accuracy: 0.6970781156827669
Precision: 0.6513098464317977
Recall: 0.533678756476684
F1-Score: 0.5866558177379985
AUC-ROC: 0.7516168382511356
Confusion Matrix:
[[1617  386]
 [ 630  721]]

 Bayesian modelling instead of grid search:
F1 scores across folds (Random Set): [0.65746673 0.67562189 0.66602223 0.67431193 0.65994236]
Mean F1 score (Random Set): 0.6666730291216855

[I 2025-01-18 14:38:13,266] A new study created in memory with name: Random Set Optimization
[I 2025-01-18 14:38:13,906] Trial 3 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 3, 'min_samples_leaf': 19}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:13,960] Trial 2 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 12}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:15,444] Trial 0 finished with value: 0.5915195190502713 and parameters: {'criterion': 'gini', 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 19}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:15,517] Trial 4 finished with value: 0.6406679008450477 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 13, 'min_samples_leaf': 11}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:15,978] Trial 1 finished with value: 0.593593741688708 and parameters: {'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 4}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:16,610] Trial 6 finished with value: 0.6511259845118981 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 6, 'min_samples_leaf': 12}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:17,013] Trial 5 finished with value: 0.6102932509545828 and parameters: {'criterion': 'entropy', 'max_depth': 16, 'min_samples_split': 19, 'min_samples_leaf': 5}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:17,174] Trial 7 finished with value: 0.641047514849393 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 14, 'min_samples_leaf': 2}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:18,543] Trial 10 finished with value: 0.6442449681997419 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 11, 'min_samples_leaf': 4}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:18,633] Trial 9 finished with value: 0.6506487425496535 and parameters: {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 15, 'min_samples_leaf': 5}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:18,679] Trial 8 finished with value: 0.6258990850507623 and parameters: {'criterion': 'entropy', 'max_depth': 13, 'min_samples_split': 18, 'min_samples_leaf': 7}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:19,342] Trial 14 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 20}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:19,435] Trial 13 finished with value: 0.6660228776792916 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 8, 'min_samples_leaf': 15}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:19,564] Trial 11 finished with value: 0.6301355109576348 and parameters: {'criterion': 'entropy', 'max_depth': 11, 'min_samples_split': 9, 'min_samples_leaf': 6}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:20,006] Trial 15 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 7, 'min_samples_leaf': 15}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:20,158] Trial 16 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 5, 'min_samples_leaf': 16}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:20,194] Trial 17 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 4, 'min_samples_leaf': 16}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:21,124] Trial 12 finished with value: 0.5906503872894742 and parameters: {'criterion': 'entropy', 'max_depth': 18, 'min_samples_split': 17, 'min_samples_leaf': 18}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:21,217] Trial 18 finished with value: 0.6517614008907825 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 16}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:21,451] Trial 20 finished with value: 0.6514756794387855 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 9}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:21,492] Trial 19 finished with value: 0.6514756794387855 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 3, 'min_samples_leaf': 9}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:22,202] Trial 21 finished with value: 0.6506419505041525 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 9}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:22,419] Trial 22 finished with value: 0.6506419505041525 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 9}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:22,434] Trial 23 finished with value: 0.6434684822000748 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 13}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:22,528] Trial 24 finished with value: 0.6431829688334961 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 20}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:23,001] Trial 25 finished with value: 0.6664168691394077 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 10, 'min_samples_leaf': 20}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:23,221] Trial 26 finished with value: 0.6664168691394077 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 4, 'min_samples_leaf': 20}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:24,489] Trial 27 finished with value: 0.6328856061772821 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 20}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:24,577] Trial 28 finished with value: 0.6284459676728243 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 18}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:25,067] Trial 29 finished with value: 0.6259885137064323 and parameters: {'criterion': 'entropy', 'max_depth': 11, 'min_samples_split': 4, 'min_samples_leaf': 18}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:25,193] Trial 30 finished with value: 0.6284459676728243 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 7, 'min_samples_leaf': 18}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:25,792] Trial 34 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 7, 'min_samples_leaf': 14}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:25,907] Trial 32 finished with value: 0.6463131768640337 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 7, 'min_samples_leaf': 18}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:26,466] Trial 35 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 13}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:26,623] Trial 36 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 3, 'min_samples_leaf': 14}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:26,727] Trial 33 finished with value: 0.6402551490903773 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 7, 'min_samples_leaf': 14}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:26,766] Trial 31 finished with value: 0.6222739248567501 and parameters: {'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 7, 'min_samples_leaf': 18}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:27,343] Trial 37 finished with value: 0.6664168691394077 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 6, 'min_samples_leaf': 11}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:27,607] Trial 38 finished with value: 0.6431153757692896 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 6, 'min_samples_leaf': 11}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:27,728] Trial 39 finished with value: 0.6434684822000748 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 6, 'min_samples_leaf': 11}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:27,761] Trial 40 finished with value: 0.6434684822000748 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 6, 'min_samples_leaf': 12}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:29,170] Trial 42 finished with value: 0.6409181276224294 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 3, 'min_samples_leaf': 17}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:29,508] Trial 41 finished with value: 0.5981445574368194 and parameters: {'criterion': 'gini', 'max_depth': 14, 'min_samples_split': 3, 'min_samples_leaf': 17}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:29,875] Trial 43 finished with value: 0.5967194749071971 and parameters: {'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 17}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:30,201] Trial 46 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 5, 'min_samples_leaf': 16}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:30,236] Trial 44 finished with value: 0.5992158189897485 and parameters: {'criterion': 'entropy', 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 16}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:30,545] Trial 47 finished with value: 0.6666730291216855 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 5, 'min_samples_leaf': 15}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:31,622] Trial 45 finished with value: 0.5904400050689385 and parameters: {'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 16}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:32,378] Trial 49 finished with value: 0.5966877114531316 and parameters: {'criterion': 'entropy', 'max_depth': 19, 'min_samples_split': 5, 'min_samples_leaf': 19}. Best is trial 3 with value: 0.6666730291216855.
[I 2025-01-18 14:38:32,431] Trial 48 finished with value: 0.596713734251351 and parameters: {'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 19}. Best is trial 3 with value: 0.6666730291216855.
Best Parameters (Random Set): {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 3, 'min_samples_leaf': 19}
Best F1 Score (Random Set): 0.6666730291216855

F1 scores across folds (Temporal Set): [0.6214797136038186, 0.6717971933001358, 0.6889714993804213, 0.6762956669498726, 0.6727423363711682]
Mean F1 score (Temporal Set): 0.6662572819210834
[I 2025-01-18 14:38:32,432] A new study created in memory with name: Temporal Set Optimization
[I 2025-01-18 14:38:33,741] Trial 0 finished with value: 0.6621871625198167 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 18, 'min_samples_leaf': 20}. Best is trial 0 with value: 0.6621871625198167.
[I 2025-01-18 14:38:34,401] Trial 3 finished with value: 0.604796486846391 and parameters: {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 16, 'min_samples_leaf': 20}. Best is trial 0 with value: 0.6621871625198167.
[I 2025-01-18 14:38:34,608] Trial 1 finished with value: 0.61856710266737 and parameters: {'criterion': 'entropy', 'max_depth': 11, 'min_samples_split': 12, 'min_samples_leaf': 10}. Best is trial 0 with value: 0.6621871625198167.
[I 2025-01-18 14:38:35,435] Trial 2 finished with value: 0.5799785965540079 and parameters: {'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 8}. Best is trial 0 with value: 0.6621871625198167.
[I 2025-01-18 14:38:35,477] Trial 6 finished with value: 0.6876090096601591 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 4, 'min_samples_leaf': 15}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:36,471] Trial 8 finished with value: 0.6867215131535104 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 4, 'min_samples_leaf': 3}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:36,473] Trial 4 finished with value: 0.5934829334755245 and parameters: {'criterion': 'entropy', 'max_depth': 17, 'min_samples_split': 5, 'min_samples_leaf': 13}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:37,070] Trial 5 finished with value: 0.580329581213921 and parameters: {'criterion': 'gini', 'max_depth': 19, 'min_samples_split': 5, 'min_samples_leaf': 8}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:37,362] Trial 7 finished with value: 0.6290484008518916 and parameters: {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 13, 'min_samples_leaf': 17}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:38,081] Trial 9 finished with value: 0.6575898294043825 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 16, 'min_samples_leaf': 2}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:38,315] Trial 12 finished with value: 0.6874506539250231 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 16, 'min_samples_leaf': 20}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:38,715] Trial 13 finished with value: 0.6847835807881943 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 8, 'min_samples_leaf': 14}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:38,885] Trial 14 finished with value: 0.6847835807881943 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 9, 'min_samples_leaf': 15}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:39,262] Trial 10 finished with value: 0.5894831169323804 and parameters: {'criterion': 'entropy', 'max_depth': 18, 'min_samples_split': 2, 'min_samples_leaf': 11}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:39,790] Trial 15 finished with value: 0.6677035307072783 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 16}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:40,109] Trial 11 finished with value: 0.6052504136309936 and parameters: {'criterion': 'entropy', 'max_depth': 16, 'min_samples_split': 11, 'min_samples_leaf': 5}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:40,176] Trial 16 finished with value: 0.6678991898101947 and parameters: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 19, 'min_samples_leaf': 17}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:40,441] Trial 17 finished with value: 0.6682065299473591 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 18}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:41,170] Trial 20 finished with value: 0.6874506539250231 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 15, 'min_samples_leaf': 18}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:41,272] Trial 18 finished with value: 0.6675257482879223 and parameters: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 14, 'min_samples_leaf': 18}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:41,506] Trial 19 finished with value: 0.6675257482879223 and parameters: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 15, 'min_samples_leaf': 18}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:42,278] Trial 24 finished with value: 0.6793321381512178 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 17, 'min_samples_leaf': 13}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:42,361] Trial 21 finished with value: 0.6415203678508925 and parameters: {'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 14, 'min_samples_leaf': 19}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:43,060] Trial 22 finished with value: 0.6367560511235816 and parameters: {'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 8, 'min_samples_leaf': 12}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:43,275] Trial 26 finished with value: 0.6876090096601591 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 15}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:43,558] Trial 23 finished with value: 0.596604293605778 and parameters: {'criterion': 'gini', 'max_depth': 14, 'min_samples_split': 8, 'min_samples_leaf': 13}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:43,979] Trial 27 finished with value: 0.6876090096601591 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 15}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:44,080] Trial 25 finished with value: 0.6377189326608921 and parameters: {'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 9, 'min_samples_leaf': 20}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:44,388] Trial 29 finished with value: 0.6793321381512178 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 10, 'min_samples_leaf': 15}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:44,621] Trial 30 finished with value: 0.6847835807881943 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 10, 'min_samples_leaf': 16}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:44,673] Trial 31 finished with value: 0.6847835807881943 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 6, 'min_samples_leaf': 16}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:45,454] Trial 28 finished with value: 0.6035316143262369 and parameters: {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 10, 'min_samples_leaf': 15}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:45,954] Trial 32 finished with value: 0.6601169044954929 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 7, 'min_samples_leaf': 9}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:46,215] Trial 33 finished with value: 0.6601169044954929 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 6, 'min_samples_leaf': 9}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:46,230] Trial 34 finished with value: 0.6601169044954929 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 12, 'min_samples_leaf': 9}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:46,897] Trial 35 finished with value: 0.659788937049669 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 12, 'min_samples_leaf': 10}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:46,926] Trial 36 finished with value: 0.6876090096601591 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 12, 'min_samples_leaf': 11}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:47,277] Trial 38 finished with value: 0.6860356098979028 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 3, 'min_samples_leaf': 11}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:47,497] Trial 37 finished with value: 0.665509426919723 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 13, 'min_samples_leaf': 11}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:48,098] Trial 40 finished with value: 0.6742196397288357 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 3, 'min_samples_leaf': 6}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:48,216] Trial 39 finished with value: 0.6742196397288357 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 3, 'min_samples_leaf': 6}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:48,291] Trial 42 finished with value: 0.6793321381512178 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 11, 'min_samples_leaf': 6}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:48,469] Trial 41 finished with value: 0.6742196397288357 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 11, 'min_samples_leaf': 6}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:48,917] Trial 43 finished with value: 0.6793321381512178 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 11, 'min_samples_leaf': 14}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:49,015] Trial 44 finished with value: 0.6793321381512178 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 11, 'min_samples_leaf': 14}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:49,279] Trial 46 finished with value: 0.6793321381512178 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 18, 'min_samples_leaf': 12}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:49,279] Trial 45 finished with value: 0.6876090096601591 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 18, 'min_samples_leaf': 14}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:49,813] Trial 48 finished with value: 0.6874506539250231 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 18, 'min_samples_leaf': 20}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:49,813] Trial 47 finished with value: 0.6876090096601591 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 17, 'min_samples_leaf': 14}. Best is trial 6 with value: 0.6876090096601591.
[I 2025-01-18 14:38:50,485] Trial 49 finished with value: 0.6477695645178063 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 6, 'min_samples_leaf': 12}. Best is trial 6 with value: 0.6876090096601591.
Best Parameters (Temporal Set): {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 4, 'min_samples_leaf': 15}
Best F1 Score (Temporal Set): 0.6876090096601591

Validation Metrics (Random Set):
Accuracy: 0.6502683363148479
Precision: 0.6147945205479453
Recall: 0.7047738693467337
F1-Score: 0.6567164179104477
AUC-ROC: 0.6710842093555176
Confusion Matrix:
[[1059  703]
 [ 470 1122]]

Validation Metrics (Temporal Set):
Accuracy: 0.6660703637447823
Precision: 0.5558949297495418
Recall: 0.6983883346124329
F1-Score: 0.6190476190476191
AUC-ROC: 0.741161958694877
Confusion Matrix:
[[1324  727]
 [ 393  910]]

Test Metrics (Random Set):
Accuracy: 0.6121049493142516
Precision: 0.5502901353965184
Recall: 0.7541418157720344
F1-Score: 0.6362873916689963
AUC-ROC: 0.6719701663550763
Confusion Matrix:
[[ 915  930]
 [ 371 1138]]

Test Metrics (Temporal Set):
Accuracy: 0.6964818127608825
Precision: 0.6543095458758109
Recall: 0.5225758697261288
F1-Score: 0.5810699588477366
AUC-ROC: 0.7427537450301231
Confusion Matrix:
[[1630  373]
 [ 645  706]]


"""