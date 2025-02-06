# Importing libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import optuna


# Load the balanced temporal training and validation datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv').values.ravel()
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv').values.ravel()
X_test_temp = pd.read_csv('X_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv').values.ravel()

# Load the balanced random training and validation datasets
X_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv').values.ravel()
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv').values.ravel()
X_test_rand = pd.read_csv('X_test_rand.csv')
y_test_rand = pd.read_csv('y_test_rand.csv').values.ravel()

# Initialize the random forest classifier with random_state for reproducibility
model_rf = RandomForestClassifier(random_state=777, n_jobs=-1)  

# Set up 5-fold cross-validation for the random set, focusing on F1 score
kf = KFold(n_splits=5, shuffle=True, random_state=777)
random_cv_scores = cross_val_score(model_rf, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')

# Print cross-validation results for the random set
print("Random Set - 5-Fold Cross-Validation F1 Scores:", random_cv_scores)
print("Random Set - Mean F1 Score:", np.mean(random_cv_scores))

# Custom function to create stratified time-series splits
def stratified_time_series_split(X, y, n_splits=5):
    indices = []
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)
    for train_index, val_index in stratified_kfold.split(X, y):
        indices.append((train_index, val_index))
    return indices

# Use the custom stratified time-series split function
stratified_splits = stratified_time_series_split(X_temp_balanced, y_temp_balanced, n_splits=5)

# Collect the F1 scores from each fold
temporal_cv_scores = []
for train_index, val_index in stratified_splits:
    X_train, X_val = X_temp_balanced.iloc[train_index], X_temp_balanced.iloc[val_index]
    y_train, y_val = y_temp_balanced[train_index], y_temp_balanced[val_index]
    
    # Fit the model and calculate F1 score on the validation set
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_val)
    f1_score_temp = f1_score(y_val, y_pred)
    temporal_cv_scores.append(f1_score_temp)

# Print the results
print("Temporal Set - Stratified Time Series Cross-Validation F1 Scores:", temporal_cv_scores)
print("Temporal Set - Mean F1 Score:", np.mean(temporal_cv_scores))

# Hyperparameter tuning using Optuna (Bayesian optimization)

def objective(trial):
    # Defining the hyperparameter space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 70, 200, step=10),
        'max_depth': trial.suggest_int('max_depth', 10, 40, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    
    model = RandomForestClassifier(**param, random_state=777, n_jobs=-1)
    
    # Cross-validation using F1 score
    kf = KFold(n_splits=5, shuffle=True, random_state=777)
    score = cross_val_score(model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1').mean()
    
    return score

# Running Optuna optimization for Random Set
study_rand = optuna.create_study(direction='maximize')
study_rand.optimize(objective, n_trials=50)

print("Best Parameters (Random Set):", study_rand.best_params)
print("Best F1 Score (Random Set):", study_rand.best_value)

# Running Optuna optimization for Temporal Set
study_temp = optuna.create_study(direction='maximize')
study_temp.optimize(objective, n_trials=50)

print("Best Parameters (Temporal Set):", study_temp.best_params)
print("Best F1 Score (Temporal Set):", study_temp.best_value)

# Visualization of Optimization History
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_rand)
plt.title('Optimization History RF (Random Set)')
plt.show()

plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_temp)
plt.title('Optimization History RF (Temporal Set)')
plt.show()

# Validation Metrics for Random Set
best_model_rand_rf = RandomForestClassifier(**study_rand.best_params, random_state=777, n_jobs=-1)
best_model_rand_rf.fit(X_rand_balanced, y_rand_balanced)
val_rand_predictions_rf = best_model_rand_rf.predict(x_val_rand)
val_rand_accuracy_rf = accuracy_score(y_val_rand, val_rand_predictions_rf)
val_rand_precision_rf = precision_score(y_val_rand, val_rand_predictions_rf)
val_rand_recall_rf = recall_score(y_val_rand, val_rand_predictions_rf)
val_rand_f1_rf = f1_score(y_val_rand, val_rand_predictions_rf)
val_rand_roc_auc_rf = roc_auc_score(y_val_rand, best_model_rand_rf.predict_proba(x_val_rand)[:, 1])
val_rand_conf_matrix_rf = confusion_matrix(y_val_rand, val_rand_predictions_rf)

print("\nValidation Metrics (Random Set):")
print(f"Accuracy: {val_rand_accuracy_rf}")
print(f"Precision: {val_rand_precision_rf}")
print(f"Recall: {val_rand_recall_rf}")
print(f"F1-Score: {val_rand_f1_rf}")
print(f"AUC-ROC: {val_rand_roc_auc_rf}")
print(f"Confusion Matrix:\n{val_rand_conf_matrix_rf}")

# Validation Metrics for Temporal Set
best_model_temp_rf = RandomForestClassifier(**study_temp.best_params, random_state=777, n_jobs=-1)
best_model_temp_rf.fit(X_temp_balanced, y_temp_balanced)
val_temp_predictions_rf = best_model_temp_rf.predict(x_val_temp)
val_temp_accuracy_rf = accuracy_score(y_val_temp, val_temp_predictions_rf)
val_temp_precision_rf = precision_score(y_val_temp, val_temp_predictions_rf)
val_temp_recall_rf = recall_score(y_val_temp, val_temp_predictions_rf)
val_temp_f1_rf = f1_score(y_val_temp, val_temp_predictions_rf)
val_temp_roc_auc_rf = roc_auc_score(y_val_temp, best_model_temp_rf.predict_proba(x_val_temp)[:, 1])
val_temp_conf_matrix_rf = confusion_matrix(y_val_temp, val_temp_predictions_rf)

print("\nValidation Metrics (Temporal Set):")
print(f"Accuracy: {val_temp_accuracy_rf}")
print(f"Precision: {val_temp_precision_rf}")
print(f"Recall: {val_temp_recall_rf}")
print(f"F1-Score: {val_temp_f1_rf}")
print(f"AUC-ROC: {val_temp_roc_auc_rf}")
print(f"Confusion Matrix:\n{val_temp_conf_matrix_rf}")

#FINAL FIT

# Combine training and validation datasets for Random Set
X_train_val_rand = pd.concat([X_rand_balanced, x_val_rand])
y_train_val_rand = np.concatenate([y_rand_balanced, y_val_rand])

# Combine training and validation datasets for Temporal Set
X_train_val_temp = pd.concat([X_temp_balanced, x_val_temp])
y_train_val_temp = np.concatenate([y_temp_balanced, y_val_temp])

# Refit the best model for the Random Set using Optuna's best params
final_model_rand_rf = RandomForestClassifier(**study_rand.best_params, random_state=777, n_jobs=-1)
final_model_rand_rf.fit(X_train_val_rand, y_train_val_rand)

# Refit the best model for the Temporal Set using Optuna's best params
final_model_temp_rf = RandomForestClassifier(**study_temp.best_params, random_state=777, n_jobs=-1)
final_model_temp_rf.fit(X_train_val_temp, y_train_val_temp)

# Permutation Importance for Random Set
perm_importance_rand_rf = permutation_importance(final_model_rand_rf, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
importance_rand_df_rf = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Importance': perm_importance_rand_rf.importances_mean
})
importance_rand_df_rf = importance_rand_df_rf[(importance_rand_df_rf['Importance'] > 0.001) | (importance_rand_df_rf['Importance'] < -0.001)] # Cut off to avoid clutter in the plot
importance_rand_df_rf.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_rand_df_rf)
plt.title('Permutation Importance RF (Random Set)')
plt.show()

# Permutation Importance for Temporal Set
perm_importance_temp_rf = permutation_importance(final_model_temp_rf, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
importance_temp_df_rf = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp_rf.importances_mean
})
importance_temp_df_rf = importance_temp_df_rf[(importance_temp_df_rf['Importance'] > 0.001) | (importance_temp_df_rf['Importance'] < -0.001)] # Cut off to avoid clutter in the plot
importance_temp_df_rf.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for Temporal Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_temp_df_rf)
plt.title('Permutation Importance RF (Temporal Set)')
plt.show()

# Final Test Set Evaluation for Random Set
test_rand_predictions_rf = final_model_rand_rf.predict(X_test_rand)

test_rand_f1_rf = f1_score(y_test_rand, test_rand_predictions_rf)
test_rand_accuracy_rf = accuracy_score(y_test_rand, test_rand_predictions_rf)
test_rand_precision_rf = precision_score(y_test_rand, test_rand_predictions_rf)
test_rand_recall_rf = recall_score(y_test_rand, test_rand_predictions_rf)
test_rand_roc_auc_rf = roc_auc_score(y_test_rand, final_model_rand_rf.predict_proba(X_test_rand)[:, 1])
test_rand_conf_matrix_rf = confusion_matrix(y_test_rand, test_rand_predictions_rf)

print("\nTest Metrics (Random Set):")
print(f"Accuracy: {test_rand_accuracy_rf}")
print(f"Precision: {test_rand_precision_rf}")
print(f"Recall: {test_rand_recall_rf}")
print(f"F1-Score: {test_rand_f1_rf}")
print(f"AUC-ROC: {test_rand_roc_auc_rf}")
print(f"Confusion Matrix:\n{test_rand_conf_matrix_rf}")

# Test Set Evaluation for Temporal Set
test_temp_predictions_rf = final_model_temp_rf.predict(X_test_temp)
test_temp_f1_rf = f1_score(y_test_temp, test_temp_predictions_rf)
test_temp_accuracy_rf = accuracy_score(y_test_temp, test_temp_predictions_rf)
test_temp_precision_rf = precision_score(y_test_temp, test_temp_predictions_rf)
test_temp_recall_rf = recall_score(y_test_temp, test_temp_predictions_rf)
test_temp_roc_auc_rf = roc_auc_score(y_test_temp, final_model_temp_rf.predict_proba(X_test_temp)[:, 1])
test_temp_conf_matrix_rf = confusion_matrix(y_test_temp, test_temp_predictions_rf)

print("\nTest Metrics (Temporal Set):")
print(f"Accuracy: {test_temp_accuracy_rf}")
print(f"Precision: {test_temp_precision_rf}")
print(f"Recall: {test_temp_recall_rf}")
print(f"F1-Score: {test_temp_f1_rf}")
print(f"AUC-ROC: {test_temp_roc_auc_rf}")
print(f"Confusion Matrix:\n{test_temp_conf_matrix_rf}")


"""
RESULTS ON ALL DATA:
Random Set - 5-Fold Cross-Validation F1 Scores: [0.81631781 0.81883899 0.81481481 0.81442851 0.81117367]
Random Set - Mean F1 Score: 0.8151147595293343

Temporal Set - Time Series Cross-Validation F1 Scores: [0.8638576  0.86434438 0.85710187 0.37896929 0.        ]
Temporal Set - Mean F1 Score: 0.5928546260213288

Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Random Set): {'n_estimators': 150, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': None, 'max_depth': None, 'criterion': 'entropy'}
Best F1 Score (Random Set): 0.6980236867024973

Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'n_estimators': 150, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 40, 'criterion': 'entropy'}
Best F1 Score (Temporal Set): 0.8043758060866548

Validation F1 Score (Random Set): 0.827489122088559
Validation F1 Score (Temporal Set): 0.6828637147786083

Validation Metrics (Random Set):
Accuracy: 0.7845268542199488
Precision: 0.8395222020254479
Recall: 0.8157961140550088
F1-Score: 0.827489122088559
AUC-ROC: 0.8842503471106868
Confusion Matrix:
[[1675  618]
 [ 730 3233]]

Validation Metrics (Temporal Set):
Accuracy: 0.6473785166240409
Precision: 0.6014180805267156
Recall: 0.7898237445959428
F1-Score: 0.6828637147786083
AUC-ROC: 0.7715723433052436
Confusion Matrix:
[[1675 1574]
 [ 632 2375]]
 
Test Metrics (Random Set):
Accuracy: 0.7906344893719035
Precision: 0.8388020833333333
Recall: 0.8233640081799591
F1-Score: 0.8310113519091846
AUC-ROC: 0.8864579381793923
[[1726  619]
 [ 691 3221]]

Test Metrics (Temporal Set):
Accuracy: 0.6469554099408662
Precision: 0.6015166340508806
Recall: 0.8091477459690688
F1-Score: 0.6900519152518592
AUC-ROC: 0.7711806286250569
[[1589 1629]
 [ 580 2459]]

 RESULTS ON DATA AFTER 2010:

Random Set - 5-Fold Cross-Validation F1 Scores: [0.62445031 0.64965398 0.64349376 0.6266433  0.62605602]
Random Set - Mean F1 Score: 0.6340594736924197
Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.647918188458729, 0.6298972993533664, 0.6866779089376055, 0.664499819429397, 0.6797927461139897]
Temporal Set - Mean F1 Score: 0.6617571924586174

Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Random Set): {'n_estimators': 150, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': None, 'max_depth': 27, 'criterion': 'gini'}
Best F1 Score (Random Set): 0.5536249014733333
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'n_estimators': 80, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 27, 'criterion': 'gini'}
Best F1 Score (Temporal Set): 0.6860192879153643

Validation Metrics (Random Set):
Accuracy: 0.7002868068833652
Precision: 0.6692586832555728
Recall: 0.6769795490298899
F1-Score: 0.6730969760166841
AUC-ROC: 0.7815735154145131
Confusion Matrix:
[[1639  638]
 [ 616 1291]]

Validation Metrics (Temporal Set):
Accuracy: 0.6336042065009561
Precision: 0.48697674418604653
Recall: 0.7088693297224103
F1-Score: 0.5773366418527709
AUC-ROC: 0.7148250767400347
Confusion Matrix:
[[1604 1103]
 [ 430 1047]]

Test Metrics (Random Set):
Accuracy: 0.6976577437858509
Precision: 0.6631578947368421
Recall: 0.6684350132625995
F1-Score: 0.665785997357992
AUC-ROC: 0.7854345621380764
Confusion Matrix:
[[1659  640]
 [ 625 1260]]

Test Metrics (Temporal Set):
Accuracy: 0.630019120458891
Precision: 0.48464007336084364
Recall: 0.7137069547602971
F1-Score: 0.5772801747678864
AUC-ROC: 0.7051579221626606
Confusion Matrix:
[[1579 1124]
 [ 424 1057]]

 after removing temporal features and adding features:
Random Set - 5-Fold Cross-Validation F1 Scores: [0.61897356 0.62146893 0.60842754 0.61256545 0.61834769]
Random Set - Mean F1 Score: 0.6159566341114386

Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.5894134477825466, 0.577755905511811, 0.644808743169399, 0.629613161405069, 0.6237942122186495]
Temporal Set - Mean F1 Score: 0.613077094017495

Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Random Set): {'n_estimators': 80, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 27, 'criterion': 'gini'}
Best F1 Score (Random Set): 0.47538775655878346

Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'n_estimators': 80, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': None, 'criterion': 'entropy'}
Best F1 Score (Temporal Set): 0.644904396952984

Validation Metrics (Random Set):
Accuracy: 0.6478831246273107
Precision: 0.6293266205160478
Recall: 0.628140703517588
F1-Score: 0.6287331027978623
AUC-ROC: 0.7090492901510961
Confusion Matrix:
[[1173  589]
 [ 592 1000]]

Validation Metrics (Temporal Set):
Accuracy: 0.6782945736434108
Precision: 0.5849772382397572
Recall: 0.5917114351496546
F1-Score: 0.588325066768409
AUC-ROC: 0.7197737434484348
Confusion Matrix:
[[1504  547]
 [ 532  771]]


Test Metrics (Random Set):
Accuracy: 0.6446034585569469
Precision: 0.6008911521323997
Recall: 0.6255798542080848
F1-Score: 0.612987012987013
AUC-ROC: 0.7037511157086389
Confusion Matrix:
[[1218  627]
 [ 565  944]]

Test Metrics (Temporal Set):
Accuracy: 0.6595110316040549
Precision: 0.5775798069784707
Recall: 0.5758697261287935
F1-Score: 0.5767234988880652
AUC-ROC: 0.6987825072162297
Confusion Matrix:
[[1434  569]
 [ 573  778]]

Bayesian modelling instead of random search:
Random Set - 5-Fold Cross-Validation F1 Scores: [0.61897356 0.62146893 0.60842754 0.61256545 0.61834769]
Random Set - Mean F1 Score: 0.6159566341114386
Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.5894134477825466, 0.577755905511811, 0.644808743169399, 0.629613161405069, 0.6237942122186495]
Temporal Set - Mean F1 Score: 0.613077094017495
[I 2025-01-18 16:56:34,938] A new study created in memory with name: no-name-5527462b-bcb1-4eb2-9b62-18ca6104c0d9
[I 2025-01-18 16:58:20,532] Trial 0 finished with value: 0.6276828296278939 and parameters: {'n_estimators': 180, 'max_depth': 25, 'min_samples_split': 9, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': None}. Best is trial 0 with value: 0.6276828296278939.
[I 2025-01-18 16:59:43,281] Trial 1 finished with value: 0.6322745514492331 and parameters: {'n_estimators': 130, 'max_depth': 20, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': None}. Best is trial 1 with value: 0.6322745514492331.
[I 2025-01-18 17:01:29,002] Trial 2 finished with value: 0.6271785733374563 and parameters: {'n_estimators': 190, 'max_depth': 40, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 1 with value: 0.6322745514492331.
[I 2025-01-18 17:01:32,492] Trial 3 finished with value: 0.6381794972965285 and parameters: {'n_estimators': 100, 'max_depth': 35, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 3 with value: 0.6381794972965285.
[I 2025-01-18 17:01:34,605] Trial 4 finished with value: 0.641717979605759 and parameters: {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 9, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 4 with value: 0.641717979605759.
[I 2025-01-18 17:03:21,497] Trial 5 finished with value: 0.6356947007708449 and parameters: {'n_estimators': 170, 'max_depth': 20, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': None}. Best is trial 4 with value: 0.641717979605759.
[I 2025-01-18 17:03:26,016] Trial 6 finished with value: 0.6396032168661193 and parameters: {'n_estimators': 150, 'max_depth': 30, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 4 with value: 0.641717979605759.
[I 2025-01-18 17:03:28,754] Trial 7 finished with value: 0.639331305138563 and parameters: {'n_estimators': 80, 'max_depth': 30, 'min_samples_split': 10, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 4 with value: 0.641717979605759.
[I 2025-01-18 17:03:31,926] Trial 8 finished with value: 0.6349842757602526 and parameters: {'n_estimators': 100, 'max_depth': 35, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 4 with value: 0.641717979605759.
[I 2025-01-18 17:03:34,491] Trial 9 finished with value: 0.639872588735788 and parameters: {'n_estimators': 80, 'max_depth': 25, 'min_samples_split': 9, 'min_samples_leaf': 4, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 4 with value: 0.641717979605759.
[I 2025-01-18 17:03:39,113] Trial 10 finished with value: 0.6523810735721185 and parameters: {'n_estimators': 130, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 10 with value: 0.6523810735721185.
[I 2025-01-18 17:03:43,657] Trial 11 finished with value: 0.6523810735721185 and parameters: {'n_estimators': 130, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 10 with value: 0.6523810735721185.
[I 2025-01-18 17:03:48,325] Trial 12 finished with value: 0.6523810735721185 and parameters: {'n_estimators': 130, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 10 with value: 0.6523810735721185.
[I 2025-01-18 17:03:56,382] Trial 13 finished with value: 0.6529765604917258 and parameters: {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:04:04,663] Trial 14 finished with value: 0.6515932712123085 and parameters: {'n_estimators': 160, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:04:12,168] Trial 15 finished with value: 0.6454901195616141 and parameters: {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:04:22,515] Trial 16 finished with value: 0.6505851956128952 and parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:04:30,390] Trial 17 finished with value: 0.6433998676310106 and parameters: {'n_estimators': 120, 'max_depth': 20, 'min_samples_split': 3, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:04:38,420] Trial 18 finished with value: 0.6512305224464833 and parameters: {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:04:42,175] Trial 19 finished with value: 0.6506238101942777 and parameters: {'n_estimators': 110, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 4, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:04:53,066] Trial 20 finished with value: 0.642271245370485 and parameters: {'n_estimators': 170, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:04:57,887] Trial 21 finished with value: 0.6523925709433189 and parameters: {'n_estimators': 140, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:05:02,745] Trial 22 finished with value: 0.6523925709433189 and parameters: {'n_estimators': 140, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:05:09,874] Trial 23 finished with value: 0.6458368278477176 and parameters: {'n_estimators': 140, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6529765604917258.
[I 2025-01-18 17:05:15,492] Trial 24 finished with value: 0.6543905199472325 and parameters: {'n_estimators': 160, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:05:23,703] Trial 25 finished with value: 0.6455991963076974 and parameters: {'n_estimators': 160, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:05:29,293] Trial 26 finished with value: 0.6523541204926718 and parameters: {'n_estimators': 170, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:05:37,830] Trial 27 finished with value: 0.6530052017668472 and parameters: {'n_estimators': 160, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:05:48,844] Trial 28 finished with value: 0.6440458372476401 and parameters: {'n_estimators': 180, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:07:38,715] Trial 29 finished with value: 0.62531800489765 and parameters: {'n_estimators': 180, 'max_depth': 25, 'min_samples_split': 3, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': None}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:07:46,792] Trial 30 finished with value: 0.6503079355702764 and parameters: {'n_estimators': 160, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:07:52,045] Trial 31 finished with value: 0.6530863077986768 and parameters: {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:08:00,027] Trial 32 finished with value: 0.6534612409300301 and parameters: {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:09:50,034] Trial 33 finished with value: 0.6309503579677845 and parameters: {'n_estimators': 160, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': None}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:09:56,580] Trial 34 finished with value: 0.6525971963066913 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:11:06,218] Trial 35 finished with value: 0.6421500955789563 and parameters: {'n_estimators': 120, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': None}. Best is trial 24 with value: 0.6543905199472325.
[I 2025-01-18 17:11:12,317] Trial 36 finished with value: 0.6546001706389968 and parameters: {'n_estimators': 170, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 36 with value: 0.6546001706389968.
[I 2025-01-18 17:11:18,460] Trial 37 finished with value: 0.6544398566588716 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 36 with value: 0.6546001706389968.
[I 2025-01-18 17:13:18,224] Trial 38 finished with value: 0.6269752858018013 and parameters: {'n_estimators': 200, 'max_depth': 40, 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': None}. Best is trial 36 with value: 0.6546001706389968.
[I 2025-01-18 17:13:21,914] Trial 39 finished with value: 0.6465919661116732 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 36 with value: 0.6546001706389968.
[I 2025-01-18 17:13:27,786] Trial 40 finished with value: 0.6539774304745058 and parameters: {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 36 with value: 0.6546001706389968.
[I 2025-01-18 17:13:33,780] Trial 41 finished with value: 0.6539774304745058 and parameters: {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 36 with value: 0.6546001706389968.
[I 2025-01-18 17:13:39,603] Trial 42 finished with value: 0.6539774304745058 and parameters: {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 36 with value: 0.6546001706389968.
[I 2025-01-18 17:13:49,929] Trial 43 finished with value: 0.6369712814191566 and parameters: {'n_estimators': 190, 'max_depth': 30, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 36 with value: 0.6546001706389968.
[I 2025-01-18 17:13:53,229] Trial 44 finished with value: 0.6461013460781755 and parameters: {'n_estimators': 170, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 36 with value: 0.6546001706389968.
[I 2025-01-18 17:13:59,408] Trial 45 finished with value: 0.6552814687215929 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 45 with value: 0.6552814687215929.
[I 2025-01-18 17:14:05,664] Trial 46 finished with value: 0.6553106795500121 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 46 with value: 0.6553106795500121.
[I 2025-01-18 17:14:11,750] Trial 47 finished with value: 0.6553106795500121 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 46 with value: 0.6553106795500121.
[I 2025-01-18 17:14:18,173] Trial 48 finished with value: 0.6346189294351128 and parameters: {'n_estimators': 200, 'max_depth': 35, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 46 with value: 0.6553106795500121.
[I 2025-01-18 17:14:24,425] Trial 49 finished with value: 0.6566792091764636 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 49 with value: 0.6566792091764636.
Best Parameters (Random Set): {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'sqrt'}  
Best F1 Score (Random Set): 0.6566792091764636

[I 2025-01-18 17:14:24,426] A new study created in memory with name: no-name-d8ab5c97-4192-4652-9e9a-f869bbc9bdac
[I 2025-01-18 17:14:32,780] Trial 0 finished with value: 0.6408496864778976 and parameters: {'n_estimators': 170, 'max_depth': 25, 'min_samples_split': 3, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:14:37,007] Trial 1 finished with value: 0.6391298028012505 and parameters: {'n_estimators': 150, 'max_depth': 25, 'min_samples_split': 7, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:15:22,881] Trial 2 finished with value: 0.6293153632104175 and parameters: {'n_estimators': 80, 'max_depth': 35, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:16:37,608] Trial 3 finished with value: 0.6247978433076659 and parameters: {'n_estimators': 110, 'max_depth': 30, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': None}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:17:47,382] Trial 4 finished with value: 0.6256080136721884 and parameters: {'n_estimators': 110, 'max_depth': 35, 'min_samples_split': 10, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': None}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:18:25,155] Trial 5 finished with value: 0.6288592592258938 and parameters: {'n_estimators': 70, 'max_depth': 40, 'min_samples_split': 8, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': None}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:18:35,445] Trial 6 finished with value: 0.6347031166575825 and parameters: {'n_estimators': 150, 'max_depth': 30, 'min_samples_split': 7, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:19:49,917] Trial 7 finished with value: 0.626919271709846 and parameters: {'n_estimators': 110, 'max_depth': 30, 'min_samples_split': 10, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': None}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:21:20,939] Trial 8 finished with value: 0.6284959482819612 and parameters: {'n_estimators': 160, 'max_depth': 40, 'min_samples_split': 3, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:22:40,517] Trial 9 finished with value: 0.6253551359550231 and parameters: {'n_estimators': 110, 'max_depth': 40, 'min_samples_split': 9, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': None}. Best is trial 0 with value: 0.6408496864778976.
[I 2025-01-18 17:22:48,967] Trial 10 finished with value: 0.6480173818176657 and parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 10 with value: 0.6480173818176657.
[I 2025-01-18 17:22:57,493] Trial 11 finished with value: 0.6480173818176657 and parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 10 with value: 0.6480173818176657.
[I 2025-01-18 17:23:03,712] Trial 12 finished with value: 0.6524500827346185 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 12 with value: 0.6524500827346185.
[I 2025-01-18 17:23:10,177] Trial 13 finished with value: 0.6532783595842314 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6532783595842314.
[I 2025-01-18 17:23:16,095] Trial 14 finished with value: 0.652145341881754 and parameters: {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6532783595842314.
[I 2025-01-18 17:23:19,823] Trial 15 finished with value: 0.645734123544455 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 13 with value: 0.6532783595842314.
[I 2025-01-18 17:23:28,200] Trial 16 finished with value: 0.6504497155411989 and parameters: {'n_estimators': 180, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6532783595842314.
[I 2025-01-18 17:23:35,000] Trial 17 finished with value: 0.6391112222210504 and parameters: {'n_estimators': 130, 'max_depth': 20, 'min_samples_split': 3, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 13 with value: 0.6532783595842314.
[I 2025-01-18 17:23:41,626] Trial 18 finished with value: 0.6556930906901711 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:23:47,486] Trial 19 finished with value: 0.642190790954493 and parameters: {'n_estimators': 180, 'max_depth': 20, 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:23:54,909] Trial 20 finished with value: 0.6402715226398318 and parameters: {'n_estimators': 130, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:24:01,542] Trial 21 finished with value: 0.6556930906901711 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:24:07,768] Trial 22 finished with value: 0.6553494324088247 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:24:15,889] Trial 23 finished with value: 0.6528091651103913 and parameters: {'n_estimators': 170, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:24:22,238] Trial 24 finished with value: 0.6553494324088247 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:24:31,324] Trial 25 finished with value: 0.6522581751793043 and parameters: {'n_estimators': 190, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:24:34,964] Trial 26 finished with value: 0.6467838834419232 and parameters: {'n_estimators': 170, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:24:43,376] Trial 27 finished with value: 0.6395226148299143 and parameters: {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:24:52,285] Trial 28 finished with value: 0.6522581751793043 and parameters: {'n_estimators': 190, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:24:57,624] Trial 29 finished with value: 0.6502181326001205 and parameters: {'n_estimators': 160, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 4, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:25:08,521] Trial 30 finished with value: 0.6440458372476401 and parameters: {'n_estimators': 180, 'max_depth': 20, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:25:14,809] Trial 31 finished with value: 0.6553494324088247 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:25:21,473] Trial 32 finished with value: 0.6556930906901711 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:25:30,890] Trial 33 finished with value: 0.6520299159327358 and parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:25:34,561] Trial 34 finished with value: 0.6467838834419232 and parameters: {'n_estimators': 170, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:25:47,043] Trial 35 finished with value: 0.6332710702982005 and parameters: {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:25:55,247] Trial 36 finished with value: 0.6504916476283948 and parameters: {'n_estimators': 180, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 4, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:26:01,635] Trial 37 finished with value: 0.6553494324088247 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:26:04,119] Trial 38 finished with value: 0.6467424445136843 and parameters: {'n_estimators': 90, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 4, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:26:13,755] Trial 39 finished with value: 0.6344972515076902 and parameters: {'n_estimators': 160, 'max_depth': 25, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:27:53,201] Trial 40 finished with value: 0.6288406267175466 and parameters: {'n_estimators': 140, 'max_depth': 35, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': None}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:27:59,668] Trial 41 finished with value: 0.6541226868499093 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:28:06,390] Trial 42 finished with value: 0.6514770575451782 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:28:12,866] Trial 43 finished with value: 0.6553494324088247 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:28:18,942] Trial 44 finished with value: 0.6550670339192346 and parameters: {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:30:11,769] Trial 45 finished with value: 0.6434766079506818 and parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': None}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:30:17,994] Trial 46 finished with value: 0.6510799811891871 and parameters: {'n_estimators': 170, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 4, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:30:24,385] Trial 47 finished with value: 0.6503874454123785 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 4, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:32:19,119] Trial 48 finished with value: 0.6452443382798309 and parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': None}. Best is trial 18 with value: 0.6556930906901711.
[I 2025-01-18 17:32:22,652] Trial 49 finished with value: 0.6518914558696274 and parameters: {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 18 with value: 0.6556930906901711.
Best Parameters (Temporal Set): {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'sqrt'}
Best F1 Score (Temporal Set): 0.6556930906901711


Validation Metrics (Random Set):
Accuracy: 0.6535480023852117
Precision: 0.6293622141997594
Recall: 0.657035175879397
F1-Score: 0.6429010448678549
AUC-ROC: 0.7201030692623162
Confusion Matrix:
[[1146  616]
 [ 546 1046]]

Validation Metrics (Temporal Set):
Accuracy: 0.6997614788312463
Precision: 0.6385767790262172
Recall: 0.523407521105142
F1-Score: 0.5752846900042177
AUC-ROC: 0.7494122441068187
Confusion Matrix:
[[1665  386]
 [ 621  682]]

Test Metrics (Random Set):
Accuracy: 0.6493738819320215
Precision: 0.6027143738433066
Recall: 0.6474486414844268
F1-Score: 0.6242811501597444
AUC-ROC: 0.7163445703376848
Confusion Matrix:
[[1201  644]
 [ 532  977]]

Test Metrics (Temporal Set):
Accuracy: 0.7197376267143709
Precision: 0.6962750716332379
Recall: 0.53960029607698
F1-Score: 0.6080066722268558
AUC-ROC: 0.7837575612894498
Confusion Matrix:
[[1685  318]
 [ 622  729]]
"""
