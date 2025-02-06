# Importing necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import optuna

# Load the balanced temporal training and validation datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')
X_test_temp = pd.read_csv('X_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv')

# Load the balanced random training and validation datasets
X_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv')
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv')
X_test_rand = pd.read_csv('X_test_rand.csv')
y_test_rand = pd.read_csv('y_test_rand.csv')


# Initialize the XGBoost classifier with randomstate 777
xgb_model = XGBClassifier(random_state=777, eval_metric='logloss') 

# Set up 5-fold cross-validation for the random set
kf = KFold(n_splits=5, shuffle=True, random_state=777)

# Perform cross-validation on the random set
random_cv_scores = cross_val_score(xgb_model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')
print("Random Set - 5-Fold Cross-Validation F1 Scores:", random_cv_scores)
print("Random Set - Mean F1 Score:", np.mean(random_cv_scores))

# Custom function to create stratified time-series splits
def stratified_time_series_split(X, y, n_splits=5):
    # Create a list of indices to hold the splits
    indices = []
    
    # Initialize the StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)

    # Generate the indices for stratified time-series split
    for train_index, val_index in stratified_kfold.split(X, y):
        # Ensure the maintainance in the temporal order
        #slice the indices based on time (first train, then test)
        indices.append((train_index, val_index))
        
    return indices

# Use the custom stratified time-series split function
stratified_splits = stratified_time_series_split(X_temp_balanced, y_temp_balanced, n_splits=5)

# Collect the F1 scores from each fold
temporal_cv_scores = []
for train_index, val_index in stratified_splits:
    X_train, X_val = X_temp_balanced.iloc[train_index], X_temp_balanced.iloc[val_index]
    y_train, y_val = y_temp_balanced.iloc[train_index], y_temp_balanced.iloc[val_index]
    
    # Fit the model and calculate F1 score on the validation set
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_val)
    
    # Calculate the F1 score for this fold
    f1_score_temp = f1_score(y_val, y_pred)
    temporal_cv_scores.append(f1_score_temp)

# Print the results
print("Temporal Set - Stratified Time Series Cross-Validation F1 Scores:", temporal_cv_scores)
print("Temporal Set - Mean F1 Score:", np.mean(temporal_cv_scores))

# Optuna optimization for XGBoost hyperparameters
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 150, 300),
        'max_depth': trial.suggest_int('max_depth', 7, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.4, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.6)
    }
    model = XGBClassifier(**param, random_state=777, eval_metric='logloss')
    cv_scores = cross_val_score(model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')

    return np.mean(cv_scores)

# Optimize hyperparameters for the Random Set
study_rand = optuna.create_study(direction='maximize')
study_rand.optimize(objective, n_trials=50, timeout=600)
print("Best Parameters (Random Set):", study_rand.best_params)
print("Best F1 Score (Random Set):", study_rand.best_value)

# Visualization of Optimization History
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_rand)
plt.title('Optimization History XGBoost (Random Set)')
plt.show()

# Retrain the model with the best parameters for the Random Set
best_model_rand = XGBClassifier(**study_rand.best_params, random_state=777, eval_metric='logloss')
best_model_rand.fit(X_rand_balanced, y_rand_balanced)

# Evaluate the best model on the validation set
val_rand_predictions = best_model_rand.predict(x_val_rand)
val_rand_accuracy = accuracy_score(y_val_rand, val_rand_predictions)
val_rand_precision = precision_score(y_val_rand, val_rand_predictions)
val_rand_recall = recall_score(y_val_rand, val_rand_predictions)
val_rand_f1 = f1_score(y_val_rand, val_rand_predictions)
val_rand_auc = roc_auc_score(y_val_rand, val_rand_predictions)

print("\nValidation Metrics (Random Set):")
print(f"Accuracy: {val_rand_accuracy}")
print(f"Precision: {val_rand_precision}")
print(f"Recall: {val_rand_recall}")
print(f"F1-Score: {val_rand_f1}")
print(f"AUC-ROC: {val_rand_auc}")

# Confusion Matrix for the Random Set
conf_matrix_rand = confusion_matrix(y_val_rand, val_rand_predictions)
print("Confusion Matrix (Random Set):")
print(conf_matrix_rand)

# Repeat the process for the Temporal Set
def objective_temp(trial):
    param = {
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 150, 300),
        'max_depth': trial.suggest_int('max_depth', 7, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.4, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.6)
    }
    model = XGBClassifier(**param, random_state=777, eval_metric='logloss')
    cv_scores = []
    for train_index, val_index in stratified_splits:
        X_train, X_val = X_temp_balanced.iloc[train_index], X_temp_balanced.iloc[val_index]
        y_train, y_val = y_temp_balanced.iloc[train_index], y_temp_balanced.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        cv_scores.append(f1_score(y_val, y_pred))
    return np.mean(cv_scores)

# Optimize hyperparameters for the Temporal Set
study_temp = optuna.create_study(direction='maximize')
study_temp.optimize(objective_temp, n_trials=50, timeout=600)
print("Best Parameters (Temporal Set):", study_temp.best_params)
print("Best F1 Score (Temporal Set):", study_temp.best_value)

# Visualization of Optimization History for Temporal Set
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_temp)
plt.title('Optimization History XGBoost (temporal Set)')
plt.show()

# Retrain the model with the best parameters for the Temporal Set
best_model_temp = XGBClassifier(**study_temp.best_params, random_state=777, eval_metric='logloss')
best_model_temp.fit(X_temp_balanced, y_temp_balanced)

# Evaluate the best model on the validation set for the Temporal Set
val_temp_predictions = best_model_temp.predict(x_val_temp)
val_temp_accuracy = accuracy_score(y_val_temp, val_temp_predictions)
val_temp_precision = precision_score(y_val_temp, val_temp_predictions)
val_temp_recall = recall_score(y_val_temp, val_temp_predictions)
val_temp_f1 = f1_score(y_val_temp, val_temp_predictions)
val_temp_auc = roc_auc_score(y_val_temp, val_temp_predictions)

print("\nValidation Metrics (Temporal Set):")
print(f"Accuracy: {val_temp_accuracy}")
print(f"Precision: {val_temp_precision}")
print(f"Recall: {val_temp_recall}")
print(f"F1-Score: {val_temp_f1}")
print(f"AUC-ROC: {val_temp_auc}")

# Confusion Matrix for the Temporal Set
conf_matrix_temp = confusion_matrix(y_val_temp, val_temp_predictions)
print("Confusion Matrix (Temporal Set):")
print(conf_matrix_temp)

# Combine training and validation datasets for Random Set
X_train_val_rand = pd.concat([X_rand_balanced, x_val_rand], axis=0)
y_train_val_rand = pd.concat([y_rand_balanced, y_val_rand], axis=0)

# Combine training and validation datasets for Temporal Set
X_train_val_temp = pd.concat([X_temp_balanced, x_val_temp], axis=0)
y_train_val_temp = pd.concat([y_temp_balanced, y_val_temp], axis=0)

# Fit final XGBoost model on the combined dataset for the Random Set
final_model_rand_xgb = XGBClassifier(**study_rand.best_params, random_state=777, eval_metric='logloss')
final_model_rand_xgb.fit(X_train_val_rand, y_train_val_rand)

# Fit final XGBoost model on the combined dataset for the Temporal Set
final_model_temp_xgb = XGBClassifier(**study_temp.best_params, random_state=777, eval_metric='logloss')
final_model_temp_xgb.fit(X_train_val_temp, y_train_val_temp)

# Test Set Evaluation for the Random Set
test_rand_predictions_xgb = final_model_rand_xgb.predict(X_test_rand)
test_rand_f1_xgb = f1_score(y_test_rand, test_rand_predictions_xgb)
test_rand_accuracy_xgb = accuracy_score(y_test_rand, test_rand_predictions_xgb)
test_rand_precision_xgb = precision_score(y_test_rand, test_rand_predictions_xgb)
test_rand_recall_xgb = recall_score(y_test_rand, test_rand_predictions_xgb)
test_rand_roc_auc_xgb = roc_auc_score(y_test_rand, final_model_rand_xgb.predict_proba(X_test_rand)[:, 1])
test_rand_conf_matrix_xgb = confusion_matrix(y_test_rand, test_rand_predictions_xgb)

print("\nTest Metrics (Random Set - XGBoost):")
print(f"Accuracy: {test_rand_accuracy_xgb}")
print(f"Precision: {test_rand_precision_xgb}")
print(f"Recall: {test_rand_recall_xgb}")
print(f"F1-Score: {test_rand_f1_xgb}")
print(f"AUC-ROC: {test_rand_roc_auc_xgb}")
print(f"Confusion Matrix:\n{test_rand_conf_matrix_xgb}")

# Test Set Evaluation for the Temporal Set
test_temp_predictions_xgb = final_model_temp_xgb.predict(X_test_temp)
test_temp_f1_xgb = f1_score(y_test_temp, test_temp_predictions_xgb)
test_temp_accuracy_xgb = accuracy_score(y_test_temp, test_temp_predictions_xgb)
test_temp_precision_xgb = precision_score(y_test_temp, test_temp_predictions_xgb)
test_temp_recall_xgb = recall_score(y_test_temp, test_temp_predictions_xgb)
test_temp_roc_auc_xgb = roc_auc_score(y_test_temp, final_model_temp_xgb.predict_proba(X_test_temp)[:, 1])
test_temp_conf_matrix_xgb = confusion_matrix(y_test_temp, test_temp_predictions_xgb)

print("\nTest Metrics (Temporal Set - XGBoost):")
print(f"Accuracy: {test_temp_accuracy_xgb}")
print(f"Precision: {test_temp_precision_xgb}")
print(f"Recall: {test_temp_recall_xgb}")
print(f"F1-Score: {test_temp_f1_xgb}")
print(f"AUC-ROC: {test_temp_roc_auc_xgb}")
print(f"Confusion Matrix:\n{test_temp_conf_matrix_xgb}")

# PERMUTATION IMPORTANCE

# Permutation Importance for the Random Set
perm_importance_rand_xgb = permutation_importance(final_model_rand_xgb, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
importance_rand_df_xgb = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Importance': perm_importance_rand_xgb.importances_mean
})

# Filter for better visualization
importance_rand_df_xgb = importance_rand_df_xgb[(importance_rand_df_xgb['Importance'] > 0.001) | (importance_rand_df_xgb['Importance'] < -0.001)]
importance_rand_df_xgb.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for the Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_rand_df_xgb)
plt.title('Permutation Importance (Random Set) - XGBoost')
plt.show()

# Permutation Importance for the Temporal Set
perm_importance_temp_xgb = permutation_importance(final_model_temp_xgb, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
importance_temp_df_xgb = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp_xgb.importances_mean
})

# Filter for better visualization
importance_temp_df_xgb = importance_temp_df_xgb[(importance_temp_df_xgb['Importance'] > 0.001) | (importance_temp_df_xgb['Importance'] < -0.001)]
importance_temp_df_xgb.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for the Temporal Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_temp_df_xgb)
plt.title('Permutation Importance (Temporal Set) - XGBoost')
plt.show()


"""

# Hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'objective': ['binary:logistic'],
    'n_estimators': [150, 170, 200, 250, 270],
    'max_depth': [ 7, 10, 12, 15, 20],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.4, 0.5, 0.6, 0.7],
    'colsample_bytree': [0.4, 0.5, 0.6, 0.7],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.4, 0.5, 0.6]
}

# RandomizedSearchCV for Random Set
print("\nFitting RandomizedSearchCV for Random Set...")
random_search_rand = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=777
)
random_search_rand.fit(X_rand_balanced, y_rand_balanced)
print("Best Parameters (Random Set):", random_search_rand.best_params_)
print("Best F1 Score (Random Set):", random_search_rand.best_score_)

# RandomizedSearchCV for Temporal Set
print("\nFitting RandomizedSearchCV for Temporal Set...")
random_search_temp = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=777
)
random_search_temp.fit(X_temp_balanced, y_temp_balanced)
print("Best Parameters (Temporal Set):", random_search_temp.best_params_)
print("Best F1 Score (Temporal Set):", random_search_temp.best_score_)

# Model Evaluation on Validation Set
# Random Set
print("\nValidation Metrics (Random Set):")
best_model_rand = random_search_rand.best_estimator_
val_rand_predictions = best_model_rand.predict(x_val_rand)
val_rand_accuracy = accuracy_score(y_val_rand, val_rand_predictions)
val_rand_precision = precision_score(y_val_rand, val_rand_predictions)
val_rand_recall = recall_score(y_val_rand, val_rand_predictions)
val_rand_f1 = f1_score(y_val_rand, val_rand_predictions)
val_rand_auc = roc_auc_score(y_val_rand, val_rand_predictions)

print(f"Accuracy: {val_rand_accuracy}")
print(f"Precision: {val_rand_precision}")
print(f"Recall: {val_rand_recall}")
print(f"F1-Score: {val_rand_f1}")
print(f"AUC-ROC: {val_rand_auc}")

# Confusion Matrix for Random Set
conf_matrix_rand = confusion_matrix(y_val_rand, val_rand_predictions)
print("Confusion Matrix (Random Set):")
print(conf_matrix_rand)

# Temporal Set
print("\nValidation Metrics (Temporal Set):")
best_model_temp = random_search_temp.best_estimator_
val_temp_predictions = best_model_temp.predict(x_val_temp)
val_temp_accuracy = accuracy_score(y_val_temp, val_temp_predictions)
val_temp_precision = precision_score(y_val_temp, val_temp_predictions)
val_temp_recall = recall_score(y_val_temp, val_temp_predictions)
val_temp_f1 = f1_score(y_val_temp, val_temp_predictions)
val_temp_auc = roc_auc_score(y_val_temp, val_temp_predictions)

print(f"Accuracy: {val_temp_accuracy}")
print(f"Precision: {val_temp_precision}")
print(f"Recall: {val_temp_recall}")
print(f"F1-Score: {val_temp_f1}")
print(f"AUC-ROC: {val_temp_auc}")

# Confusion Matrix for Temporal Set
conf_matrix_temp = confusion_matrix(y_val_temp, val_temp_predictions)
print("Confusion Matrix (Temporal Set):")
print(conf_matrix_temp)


# Combine training and validation datasets for Random Set
X_train_val_rand = pd.concat([X_rand_balanced, x_val_rand], axis=0)
y_train_val_rand = pd.concat([y_rand_balanced, y_val_rand], axis=0)

# Combine training and validation datasets for Temporal Set
X_train_val_temp = pd.concat([X_temp_balanced, x_val_temp], axis=0)
y_train_val_temp = pd.concat([y_temp_balanced, y_val_temp], axis=0)

# Fit final XGBoost model on the combined dataset for the Random Set
final_model_rand_xgb = XGBClassifier(**random_search_rand.best_params_, random_state=777, eval_metric='logloss')
final_model_rand_xgb.fit(X_train_val_rand, y_train_val_rand)

# Fit final XGBoost model on the combined dataset for the Temporal Set
final_model_temp_xgb = XGBClassifier(**random_search_temp.best_params_, random_state=777, eval_metric='logloss')
final_model_temp_xgb.fit(X_train_val_temp, y_train_val_temp)

#PERMUTATION IMPORTANCE

# Permutation Importance for the Random Set
perm_importance_rand_xgb = permutation_importance(final_model_rand_xgb, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
importance_rand_df_xgb = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Importance': perm_importance_rand_xgb.importances_mean
})

#Using the same cutoff as with the other models to avoid clutter in the plot
importance_rand_df_xgb = importance_rand_df_xgb[(importance_rand_df_xgb['Importance'] > 0.001) | (importance_rand_df_xgb['Importance'] < -0.001)]  # Filter for better visualization
importance_rand_df_xgb.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for the Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_rand_df_xgb)
plt.title('Permutation Importance (Random Set) - XGBoost')
plt.show()

# Permutation Importance for the Temporal Set
perm_importance_temp_xgb = permutation_importance(final_model_temp_xgb, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
importance_temp_df_xgb = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp_xgb.importances_mean
})

#Using the same cutoff as with the other models to avoid clutter in the plot
importance_temp_df_xgb = importance_temp_df_xgb[(importance_temp_df_xgb['Importance'] > 0.001) | (importance_temp_df_xgb['Importance'] < -0.001)]
importance_temp_df_xgb.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for the Temporal Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_temp_df_xgb)
plt.title('Permutation Importance (Temporal Set) - XGBoost')
plt.show()


# TEST SET EVALUATION 

# Test Set Evaluation for the Random Set
test_rand_predictions_xgb = final_model_rand_xgb.predict(X_test_rand)
test_rand_f1_xgb = f1_score(y_test_rand, test_rand_predictions_xgb)
test_rand_accuracy_xgb = accuracy_score(y_test_rand, test_rand_predictions_xgb)
test_rand_precision_xgb = precision_score(y_test_rand, test_rand_predictions_xgb)
test_rand_recall_xgb = recall_score(y_test_rand, test_rand_predictions_xgb)
test_rand_roc_auc_xgb = roc_auc_score(y_test_rand, final_model_rand_xgb.predict_proba(X_test_rand)[:, 1])
test_rand_conf_matrix_xgb = confusion_matrix(y_test_rand, test_rand_predictions_xgb)

print("\nTest Metrics (Random Set - XGBoost):")
print(f"Accuracy: {test_rand_accuracy_xgb}")
print(f"Precision: {test_rand_precision_xgb}")
print(f"Recall: {test_rand_recall_xgb}")
print(f"F1-Score: {test_rand_f1_xgb}")
print(f"AUC-ROC: {test_rand_roc_auc_xgb}")
print(f"Confusion Matrix:\n{test_rand_conf_matrix_xgb}")

# Test Set Evaluation for the Temporal Set
test_temp_predictions_xgb = final_model_temp_xgb.predict(X_test_temp)
test_temp_f1_xgb = f1_score(y_test_temp, test_temp_predictions_xgb)
test_temp_accuracy_xgb = accuracy_score(y_test_temp, test_temp_predictions_xgb)
test_temp_precision_xgb = precision_score(y_test_temp, test_temp_predictions_xgb)
test_temp_recall_xgb = recall_score(y_test_temp, test_temp_predictions_xgb)
test_temp_roc_auc_xgb = roc_auc_score(y_test_temp, final_model_temp_xgb.predict_proba(X_test_temp)[:, 1])
test_temp_conf_matrix_xgb = confusion_matrix(y_test_temp, test_temp_predictions_xgb)

print("\nTest Metrics (Temporal Set - XGBoost):")
print(f"Accuracy: {test_temp_accuracy_xgb}")
print(f"Precision: {test_temp_precision_xgb}")
print(f"Recall: {test_temp_recall_xgb}")
print(f"F1-Score: {test_temp_f1_xgb}")
print(f"AUC-ROC: {test_temp_roc_auc_xgb}")
print(f"Confusion Matrix:\n{test_temp_conf_matrix_xgb}")

"""

"""METRICS ON ALL DATA

Random Set - 5-Fold Cross-Validation F1 Scores: [0.81913303 0.82719547 0.82200087 0.81839878 0.81979257]
Random Set - Mean F1 Score: 0.8213041439894152

Temporal Set - Time Series Cross-Validation F1 Scores: [0.85541126 0.86601775 0.85850144 0.32064985 0.        ]
Temporal Set - Mean F1 Score: 0.5801160592964267

Fitting RandomizedSearchCV for Random Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Random Set): {'subsample': 0.4, 'n_estimators': 270, 'min_child_weight': 7, 'max_depth': 10, 'learning_rate': 0.01, 'gamma': 0.4, 'colsample_bytree': 0.4}
Best F1 Score (Random Set): 0.7166564492146653

Fitting RandomizedSearchCV for Temporal Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'subsample': 0.7, 'n_estimators': 200, 'min_child_weight': 1, 'max_depth': 20, 'learning_rate': 0.2, 'gamma': 0.4, 'colsample_bytree': 0.5}
Best F1 Score (Temporal Set): 0.8038081133626335

Validation Metrics (Random Set):
Accuracy: 0.782608695652174
Precision: 0.8555039606664846
Recall: 0.7903103709311128
F1-Score: 0.8216159496327387
AUC-ROC: 0.7798041169963021
Confusion Matrix (Random Set):
[[1764  529]
 [ 831 3132]]

Validation Metrics (Temporal Set):
Accuracy: 0.6150895140664961
Precision: 0.5678062033054109
Recall: 0.8340538742933156
F1-Score: 0.6756465517241379
AUC-ROC: 0.6232442347766978
Confusion Matrix (Temporal Set):
[[1340 1909]
 [ 499 2508]]
 
 Test Metrics (Random Set - XGBoost):
Accuracy: 0.797027329391082
Precision: 0.862912087912088
Recall: 0.8029141104294478
F1-Score: 0.8318326271186441
AUC-ROC: 0.8955962954726804

Test Metrics (Temporal Set - XGBoost):
Accuracy: 0.6205849448617549
Precision: 0.574635241301908
Recall: 0.8423823626192827
F1-Score: 0.6832132372564719
AUC-ROC: 0.7513716444866005

METRICS ON DATA AFTER 2010

Random Set - 5-Fold Cross-Validation F1 Scores: [0.66403855 0.64892704 0.66223404 0.64736387 0.63555556]
Random Set - Mean F1 Score: 0.651623810918287
Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.621765601217656, 0.6408368849283224, 0.673233695652174, 0.6735640385301463, 0.6851211072664359]
Temporal Set - Mean F1 Score: 0.6589042655189469

Fitting RandomizedSearchCV for Random Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Random Set): {'subsample': 0.4, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.4}
Best F1 Score (Random Set): 0.5162034835411139

Fitting RandomizedSearchCV for Temporal Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'subsample': 0.4, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.4}
Best F1 Score (Temporal Set): 0.6891833090566674

Validation Metrics (Random Set):
Accuracy: 0.7069789674952199
Precision: 0.6717095310136157
Recall: 0.6984792868379653
F1-Score: 0.6848329048843187
AUC-ROC: 0.7062883917720788
Confusion Matrix (Random Set):
[[1626  651]
 [ 575 1332]]

Validation Metrics (Temporal Set):
Accuracy: 0.6816443594646272
Precision: 0.5386254661694193
Recall: 0.6844955991875423
F1-Score: 0.6028622540250448
AUC-ROC: 0.6822921291098406
Confusion Matrix (Temporal Set):
[[1841  866]
 [ 466 1011]]

Test Metrics (Random Set - XGBoost):
Accuracy: 0.6928776290630975
Precision: 0.6592356687898089
Recall: 0.6588859416445624
F1-Score: 0.6590607588219688
AUC-ROC: 0.7841517993638105
Confusion Matrix:
[[1657  642]
 [ 643 1242]]

Test Metrics (Temporal Set - XGBoost):
Accuracy: 0.6663479923518164
Precision: 0.5217169136433316
Recall: 0.6893990546927752
F1-Score: 0.5939499709133217
AUC-ROC: 0.746491444347604
Confusion Matrix:
[[1767  936]
 [ 460 1021]]

  After removing temporal features and adding in new features to mitigate bias:

Random Set - 5-Fold Cross-Validation F1 Scores: [0.62655602 0.62668046 0.61182519 0.61363636 0.63631765]
Random Set - Mean F1 Score: 0.6230031362122106
Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.5772277227722773, 0.6060898985016917, 0.6523206751054852, 0.6517739816031538, 0.6394678492239466]
Temporal Set - Mean F1 Score: 0.6253760254413109

Fitting RandomizedSearchCV for Random Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Random Set): {'subsample': 0.4, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.4}
Best F1 Score (Random Set): 0.48574747510629057

Fitting RandomizedSearchCV for Temporal Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'subsample': 0.4, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.4}
Best F1 Score (Temporal Set): 0.6644382184524034

Validation Metrics (Random Set):
Accuracy: 0.6645796064400715
Precision: 0.6419452887537994
Recall: 0.6633165829145728
F1-Score: 0.6524559777571826
AUC-ROC: 0.6645186773823716
Confusion Matrix (Random Set):
[[1173  589]
 [ 536 1056]]

Validation Metrics (Temporal Set):
Accuracy: 0.7003577817531306
Precision: 0.626057529610829
Recall: 0.56792018419033
F1-Score: 0.5955734406438632
AUC-ROC: 0.6762077761517228
Confusion Matrix (Temporal Set):
[[1609  442]
 [ 563  740]]

Test Metrics (Random Set - XGBoost):
Accuracy: 0.6583184257602862
Precision: 0.6136505948653725
Recall: 0.6494367130550033
F1-Score: 0.6310367031551835
AUC-ROC: 0.7191097318527857
Confusion Matrix:
[[1228  617]
 [ 529  980]]

Test Metrics (Temporal Set - XGBoost):
Accuracy: 0.714370900417412
Precision: 0.6880382775119617
Recall: 0.5321983715766099
F1-Score: 0.6001669449081802
AUC-ROC: 0.7794043575643197
Confusion Matrix:
[[1677  326]
 [ 632  719]]

 Baysian optimization instead of random search:


Random Set - 5-Fold Cross-Validation F1 Scores: [0.62655602 0.62668046 0.61182519 0.61363636 0.63631765]
Random Set - Mean F1 Score: 0.6230031362122106

Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.5772277227722773, 0.6060898985016917, 0.6523206751054852, 0.6517739816031538, 0.6394678492239466]
Temporal Set - Mean F1 Score: 0.6253760254413109

[I 2025-01-18 19:24:17,847] A new study created in memory with name: no-name-51fc7a27-2275-4f3a-99a9-1f88b828f7ac
[I 2025-01-18 19:24:27,161] Trial 0 finished with value: 0.6233222616511541 and parameters: {'n_estimators': 176, 'max_depth': 17, 'learning_rate': 0.09680967651021592, 'subsample': 0.7114794698833227, 'colsample_bytree': 0.6349144473148846, 'min_child_weight': 7, 'gamma': 0.5114367283313451}. Best is trial 0 with value: 0.6233222616511541.
[I 2025-01-18 19:24:37,176] Trial 1 finished with value: 0.620210071550256 and parameters: {'n_estimators': 230, 'max_depth': 14, 'learning_rate': 0.1068910925677635, 'subsample': 0.7793417094816635, 'colsample_bytree': 0.4359630116795752, 'min_child_weight': 8, 'gamma': 0.5409425445448419}. Best is trial 0 with value: 0.6233222616511541.
[I 2025-01-18 19:24:49,767] Trial 2 finished with value: 0.6107081344002367 and parameters: {'n_estimators': 239, 'max_depth': 16, 'learning_rate': 0.14969117207328517, 'subsample': 0.5857694263242909, 'colsample_bytree': 0.5347213413203149, 'min_child_weight': 4, 'gamma': 0.55769374737336}. Best is trial 0 with value: 0.6233222616511541.
[I 2025-01-18 19:24:59,145] Trial 3 finished with value: 0.609857742973045 and parameters: {'n_estimators': 173, 'max_depth': 12, 'learning_rate': 0.16805421478405294, 'subsample': 0.6508142140424853, 'colsample_bytree': 0.6278153070694699, 'min_child_weight': 4, 'gamma': 0.210754490682688}. Best is trial 0 with value: 0.6233222616511541.
[I 2025-01-18 19:25:07,721] Trial 4 finished with value: 0.6306202686862213 and parameters: {'n_estimators': 202, 'max_depth': 10, 'learning_rate': 0.06683990239874206, 'subsample': 0.6891683148243377, 'colsample_bytree': 0.7749570416549743, 'min_child_weight': 9, 'gamma': 0.427012488172511}. Best is trial 4 with value: 0.6306202686862213.
[I 2025-01-18 19:25:25,844] Trial 5 finished with value: 0.6152595190356256 and parameters: {'n_estimators': 179, 'max_depth': 18, 'learning_rate': 0.09100252897104126, 'subsample': 0.5379686861148611, 'colsample_bytree': 0.453748650047528, 'min_child_weight': 1, 'gamma': 0.1321671632044405}. Best is trial 4 with value: 0.6306202686862213.
[I 2025-01-18 19:25:37,399] Trial 6 finished with value: 0.6047944465307665 and parameters: {'n_estimators': 249, 'max_depth': 13, 'learning_rate': 0.2885190608388238, 'subsample': 0.5325127168380244, 'colsample_bytree': 0.7266547095563295, 'min_child_weight': 5, 'gamma': 0.017417959659646275}. Best is trial 4 with value: 0.6306202686862213.
[I 2025-01-18 19:25:46,838] Trial 7 finished with value: 0.6048445147624835 and parameters: {'n_estimators': 231, 'max_depth': 13, 'learning_rate': 0.2621321449383274, 'subsample': 0.6378094307635674, 'colsample_bytree': 0.6210426921024605, 'min_child_weight': 9, 'gamma': 0.12883924958990614}. Best is trial 4 with value: 0.6306202686862213.
[I 2025-01-18 19:25:58,055] Trial 8 finished with value: 0.6060110327075414 and parameters: {'n_estimators': 270, 'max_depth': 13, 'learning_rate': 0.22955502194532998, 'subsample': 0.6538862307257312, 'colsample_bytree': 0.5836523862962161, 'min_child_weight': 7, 'gamma': 0.2943951611846701}. Best is trial 4 with value: 0.6306202686862213.
[I 2025-01-18 19:26:08,946] Trial 9 finished with value: 0.6122299679085732 and parameters: {'n_estimators': 250, 'max_depth': 11, 'learning_rate': 0.1322057495271659, 'subsample': 0.4612349519098861, 'colsample_bytree': 0.7914876543601971, 'min_child_weight': 5, 'gamma': 0.1102881717235596}. Best is trial 4 with value: 0.6306202686862213.
[I 2025-01-18 19:26:16,048] Trial 10 finished with value: 0.6557088368913581 and parameters: {'n_estimators': 198, 'max_depth': 7, 'learning_rate': 0.012574724239323078, 'subsample': 0.7937594696860386, 'colsample_bytree': 0.7247254129878251, 'min_child_weight': 10, 'gamma': 0.4166471476097849}. Best is trial 10 with value: 0.6557088368913581.
[I 2025-01-18 19:26:23,069] Trial 11 finished with value: 0.6572570165644214 and parameters: {'n_estimators': 202, 'max_depth': 7, 'learning_rate': 0.01392099508269402, 'subsample': 0.7918155280974408, 'colsample_bytree': 0.7237780458877486, 'min_child_weight': 10, 'gamma': 0.39922715745901716}. Best is trial 11 with value: 0.6572570165644214.
[I 2025-01-18 19:26:29,844] Trial 12 finished with value: 0.6548739976660614 and parameters: {'n_estimators': 201, 'max_depth': 7, 'learning_rate': 0.017898406751729136, 'subsample': 0.7994998986985604, 'colsample_bytree': 0.7029457816359036, 'min_child_weight': 10, 'gamma': 0.38485488775140597}. Best is trial 11 with value: 0.6572570165644214.
[I 2025-01-18 19:26:36,854] Trial 13 finished with value: 0.6573016446848662 and parameters: {'n_estimators': 202, 'max_depth': 7, 'learning_rate': 0.013063707584806348, 'subsample': 0.7415643311748595, 'colsample_bytree': 0.7177981964287334, 'min_child_weight': 10, 'gamma': 0.40264314414263597}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:26:47,451] Trial 14 finished with value: 0.6305055331277802 and parameters: {'n_estimators': 295, 'max_depth': 9, 'learning_rate': 0.051821365361983784, 'subsample': 0.7289191364240469, 'colsample_bytree': 0.6906621945172486, 'min_child_weight': 7, 'gamma': 0.32408196226603925}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:26:54,897] Trial 15 finished with value: 0.616108067315478 and parameters: {'n_estimators': 156, 'max_depth': 20, 'learning_rate': 0.19266319624260334, 'subsample': 0.7471813122300863, 'colsample_bytree': 0.5428856136241892, 'min_child_weight': 10, 'gamma': 0.48494778399698124}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:27:03,366] Trial 16 finished with value: 0.6373898528179092 and parameters: {'n_estimators': 211, 'max_depth': 8, 'learning_rate': 0.048680030395562054, 'subsample': 0.7473702297693934, 'colsample_bytree': 0.6737594259735666, 'min_child_weight': 2, 'gamma': 0.33579276082222154}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:27:11,376] Trial 17 finished with value: 0.6399319327677395 and parameters: {'n_estimators': 216, 'max_depth': 9, 'learning_rate': 0.029762524979192115, 'subsample': 0.4084411849674045, 'colsample_bytree': 0.7581141100012091, 'min_child_weight': 8, 'gamma': 0.24461349369235244}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:27:18,844] Trial 18 finished with value: 0.6316574379444437 and parameters: {'n_estimators': 186, 'max_depth': 10, 'learning_rate': 0.07026392835854053, 'subsample': 0.6918117983788608, 'colsample_bytree': 0.6753921456210602, 'min_child_weight': 9, 'gamma': 0.46737526278572616}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:27:24,125] Trial 19 finished with value: 0.6298326039933427 and parameters: {'n_estimators': 158, 'max_depth': 7, 'learning_rate': 0.12358502313640042, 'subsample': 0.5997427379840028, 'colsample_bytree': 0.747783108252208, 'min_child_weight': 6, 'gamma': 0.5918380292664894}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:27:34,607] Trial 20 finished with value: 0.609413253960775 and parameters: {'n_estimators': 215, 'max_depth': 15, 'learning_rate': 0.2062819816550996, 'subsample': 0.7780914384245111, 'colsample_bytree': 0.7994247396734617, 'min_child_weight': 8, 'gamma': 0.3597093546757106}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:27:41,560] Trial 21 finished with value: 0.6542184267812474 and parameters: {'n_estimators': 193, 'max_depth': 7, 'learning_rate': 0.010166883692098953, 'subsample': 0.7656314818365598, 'colsample_bytree': 0.7169592458173957, 'min_child_weight': 10, 'gamma': 0.4189657768362913}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:27:48,283] Trial 22 finished with value: 0.6474568893460322 and parameters: {'n_estimators': 199, 'max_depth': 8, 'learning_rate': 0.03569626662519322, 'subsample': 0.793257085210157, 'colsample_bytree': 0.6563146515509173, 'min_child_weight': 10, 'gamma': 0.4312070783192638}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:27:54,715] Trial 23 finished with value: 0.6356159061279143 and parameters: {'n_estimators': 167, 'max_depth': 9, 'learning_rate': 0.0712456501575145, 'subsample': 0.7190422098000566, 'colsample_bytree': 0.7412995716647625, 'min_child_weight': 9, 'gamma': 0.38852041807878146}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:28:02,104] Trial 24 finished with value: 0.653532176708616 and parameters: {'n_estimators': 187, 'max_depth': 8, 'learning_rate': 0.010306097856547104, 'subsample': 0.7399775522723178, 'colsample_bytree': 0.5753599788193301, 'min_child_weight': 10, 'gamma': 0.27373423691133425}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:28:11,985] Trial 25 finished with value: 0.6342731997292218 and parameters: {'n_estimators': 211, 'max_depth': 11, 'learning_rate': 0.041465346819989764, 'subsample': 0.6846897832000529, 'colsample_bytree': 0.7059814923622489, 'min_child_weight': 9, 'gamma': 0.4663056994161922}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:28:18,649] Trial 26 finished with value: 0.6342947554497419 and parameters: {'n_estimators': 219, 'max_depth': 7, 'learning_rate': 0.08105758196756381, 'subsample': 0.7661348288119917, 'colsample_bytree': 0.7623661678091526, 'min_child_weight': 8, 'gamma': 0.37921936147285107}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:28:26,710] Trial 27 finished with value: 0.6455053075344213 and parameters: {'n_estimators': 191, 'max_depth': 10, 'learning_rate': 0.02880329627524613, 'subsample': 0.7925481071881201, 'colsample_bytree': 0.6532158171724384, 'min_child_weight': 10, 'gamma': 0.5123412857009345}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:28:33,986] Trial 28 finished with value: 0.635920236466301 and parameters: {'n_estimators': 226, 'max_depth': 8, 'learning_rate': 0.05732305183814751, 'subsample': 0.756399865736186, 'colsample_bytree': 0.4776305319626787, 'min_child_weight': 6, 'gamma': 0.1910992441566358}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:28:41,701] Trial 29 finished with value: 0.626656510667229 and parameters: {'n_estimators': 176, 'max_depth': 11, 'learning_rate': 0.1027091637603992, 'subsample': 0.7143405728429849, 'colsample_bytree': 0.6064126949906096, 'min_child_weight': 7, 'gamma': 0.33449034760122337}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:28:50,566] Trial 30 finished with value: 0.6424972091273226 and parameters: {'n_estimators': 242, 'max_depth': 9, 'learning_rate': 0.029686121125468982, 'subsample': 0.6293384061479845, 'colsample_bytree': 0.7298226090386476, 'min_child_weight': 9, 'gamma': 0.43019470220135847}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:28:57,725] Trial 31 finished with value: 0.656834296686405 and parameters: {'n_estimators': 201, 'max_depth': 7, 'learning_rate': 0.011351491744269944, 'subsample': 0.7934085465822966, 'colsample_bytree': 0.6981273142093523, 'min_child_weight': 10, 'gamma': 0.38199471582047106}. Best is trial 13 with value: 0.6573016446848662.
[I 2025-01-18 19:29:04,910] Trial 32 finished with value: 0.6578463852922667 and parameters: {'n_estimators': 203, 'max_depth': 7, 'learning_rate': 0.014561385799380593, 'subsample': 0.7702827350963908, 'colsample_bytree': 0.6817148840186857, 'min_child_weight': 10, 'gamma': 0.5149449757160774}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:29:13,630] Trial 33 finished with value: 0.6379192859000937 and parameters: {'n_estimators': 225, 'max_depth': 8, 'learning_rate': 0.04578195857859365, 'subsample': 0.7675767327021733, 'colsample_bytree': 0.6487406524651428, 'min_child_weight': 8, 'gamma': 0.5326868890190332}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:29:20,229] Trial 34 finished with value: 0.6528025268712803 and parameters: {'n_estimators': 205, 'max_depth': 7, 'learning_rate': 0.02802404357776666, 'subsample': 0.6990329674205565, 'colsample_bytree': 0.6759872458638866, 'min_child_weight': 9, 'gamma': 0.5831171772749635}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:29:27,894] Trial 35 finished with value: 0.6371964991554694 and parameters: {'n_estimators': 182, 'max_depth': 9, 'learning_rate': 0.058947670978046166, 'subsample': 0.6681140510106394, 'colsample_bytree': 0.40465288594625604, 'min_child_weight': 3, 'gamma': 0.4899378801309723}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:29:36,627] Trial 36 finished with value: 0.6225457029325542 and parameters: {'n_estimators': 169, 'max_depth': 18, 'learning_rate': 0.09065101401329151, 'subsample': 0.723918846820466, 'colsample_bytree': 0.6954930892048025, 'min_child_weight': 10, 'gamma': 0.5250660876391932}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:29:44,627] Trial 37 finished with value: 0.6166740757551411 and parameters: {'n_estimators': 205, 'max_depth': 10, 'learning_rate': 0.15385338879757834, 'subsample': 0.775404194289964, 'colsample_bytree': 0.6346126365416785, 'min_child_weight': 9, 'gamma': 0.4627622816741752}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:29:52,811] Trial 38 finished with value: 0.6315376733373961 and parameters: {'n_estimators': 235, 'max_depth': 8, 'learning_rate': 0.07786370100031213, 'subsample': 0.5446109542230481, 'colsample_bytree': 0.7668167175617443, 'min_child_weight': 8, 'gamma': 0.2744807157931557}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:30:02,630] Trial 39 finished with value: 0.6423791323679561 and parameters: {'n_estimators': 194, 'max_depth': 12, 'learning_rate': 0.024065710168225406, 'subsample': 0.7456810274241606, 'colsample_bytree': 0.6851746327925429, 'min_child_weight': 9, 'gamma': 0.35743565399197985}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:30:12,909] Trial 40 finished with value: 0.6124204591508592 and parameters: {'n_estimators': 224, 'max_depth': 14, 'learning_rate': 0.11525269291056718, 'subsample': 0.6696992571780368, 'colsample_bytree': 0.7382732100275716, 'min_child_weight': 10, 'gamma': 0.4462888049402273}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:30:19,947] Trial 41 finished with value: 0.6575494602429123 and parameters: {'n_estimators': 197, 'max_depth': 7, 'learning_rate': 0.012337942801973286, 'subsample': 0.7950139342887712, 'colsample_bytree': 0.717805247576244, 'min_child_weight': 10, 'gamma': 0.39320892961327397}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:30:26,668] Trial 42 finished with value: 0.6464792418773221 and parameters: {'n_estimators': 209, 'max_depth': 7, 'learning_rate': 0.04143595315660709, 'subsample': 0.778989507596592, 'colsample_bytree': 0.7789672642168918, 'min_child_weight': 10, 'gamma': 0.5558293693998149}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:30:33,741] Trial 43 finished with value: 0.6536345156575024 and parameters: {'n_estimators': 178, 'max_depth': 8, 'learning_rate': 0.02014754090862779, 'subsample': 0.7973541186839712, 'colsample_bytree': 0.7139950902968805, 'min_child_weight': 9, 'gamma': 0.4002705211964196}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:30:39,956] Trial 44 finished with value: 0.6440761295036228 and parameters: {'n_estimators': 198, 'max_depth': 7, 'learning_rate': 0.05789534958131394, 'subsample': 0.733671586298118, 'colsample_bytree': 0.6575985751738362, 'min_child_weight': 10, 'gamma': 0.31392682142187}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:30:48,203] Trial 45 finished with value: 0.650999886667352 and parameters: {'n_estimators': 185, 'max_depth': 9, 'learning_rate': 0.013448569325639323, 'subsample': 0.7608635962257946, 'colsample_bytree': 0.6146021848656793, 'min_child_weight': 9, 'gamma': 0.4988304221823563}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:31:00,103] Trial 46 finished with value: 0.6070076021483726 and parameters: {'n_estimators': 206, 'max_depth': 16, 'learning_rate': 0.17496124062908053, 'subsample': 0.7801572379491243, 'colsample_bytree': 0.7104339025353884, 'min_child_weight': 4, 'gamma': 0.3481163332205025}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:31:06,804] Trial 47 finished with value: 0.6093672067901146 and parameters: {'n_estimators': 219, 'max_depth': 7, 'learning_rate': 0.29115480074479444, 'subsample': 0.5546795413845411, 'colsample_bytree': 0.7804265671337303, 'min_child_weight': 10, 'gamma': 0.4040820499070873}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:31:18,008] Trial 48 finished with value: 0.6340305674571068 and parameters: {'n_estimators': 256, 'max_depth': 8, 'learning_rate': 0.03972918818573017, 'subsample': 0.47305943940981376, 'colsample_bytree': 0.7464193287406621, 'min_child_weight': 1, 'gamma': 0.3782615456235931}. Best is trial 32 with value: 0.6578463852922667.
[I 2025-01-18 19:31:25,651] Trial 49 finished with value: 0.6133761986620248 and parameters: {'n_estimators': 192, 'max_depth': 10, 'learning_rate': 0.2507330948481575, 'subsample': 0.7015846716739306, 'colsample_bytree': 0.7217135034374099, 'min_child_weight': 10, 'gamma': 0.0284036647790058}. Best is trial 32 with value: 0.6578463852922667.
Best Parameters (Random Set): {'n_estimators': 203, 'max_depth': 7, 'learning_rate': 0.014561385799380593, 'subsample': 0.7702827350963908, 'colsample_bytree': 0.6817148840186857, 'min_child_weight': 10, 'gamma': 0.5149449757160774}
Best F1 Score (Random Set): 0.6578463852922667


Validation Metrics (Random Set):
Accuracy: 0.669051878354204
Precision: 0.6415981198589894
Recall: 0.6859296482412061
F1-Score: 0.663023679417122
AUC-ROC: 0.6698660727017608
Confusion Matrix (Random Set):
[[1152  610]
 [ 500 1092]]

[I 2025-01-18 19:31:44,398] A new study created in memory with name: no-name-a568696d-4df6-4b93-80e8-79abeace3ca3
[I 2025-01-18 19:31:49,551] Trial 0 finished with value: 0.6240724374748746 and parameters: {'n_estimators': 157, 'max_depth': 8, 'learning_rate': 0.14348900346599253, 'subsample': 0.4335836956207625, 'colsample_bytree': 0.5577247078151476, 'min_child_weight': 9, 'gamma': 0.4418039339905513}. Best is trial 0 with value: 0.6240724374748746.
[I 2025-01-18 19:31:57,389] Trial 1 finished with value: 0.6135792462385263 and parameters: {'n_estimators': 235, 'max_depth': 9, 'learning_rate': 0.19076170746758042, 'subsample': 0.7814716007429291, 'colsample_bytree': 0.4889420422658717, 'min_child_weight': 3, 'gamma': 0.5979610153871961}. Best is trial 0 with value: 0.6240724374748746.
[I 2025-01-18 19:32:03,287] Trial 2 finished with value: 0.647567599436012 and parameters: {'n_estimators': 206, 'max_depth': 7, 'learning_rate': 0.05042211531927783, 'subsample': 0.6879665035956983, 'colsample_bytree': 0.40041107586940816, 'min_child_weight': 7, 'gamma': 0.17032260105216102}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:32:14,983] Trial 3 finished with value: 0.6125310032945598 and parameters: {'n_estimators': 216, 'max_depth': 17, 'learning_rate': 0.15174652244306427, 'subsample': 0.6702976427693752, 'colsample_bytree': 0.6741384124455014, 'min_child_weight': 6, 'gamma': 0.376370283281616}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:32:22,313] Trial 4 finished with value: 0.6260346561528637 and parameters: {'n_estimators': 183, 'max_depth': 8, 'learning_rate': 0.08030153753549292, 'subsample': 0.7591871326631487, 'colsample_bytree': 0.6881229750319895, 'min_child_weight': 2, 'gamma': 0.2903709072206108}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:32:36,449] Trial 5 finished with value: 0.6003461724437864 and parameters: {'n_estimators': 244, 'max_depth': 14, 'learning_rate': 0.29077642399329634, 'subsample': 0.49618431313032163, 'colsample_bytree': 0.503019797112359, 'min_child_weight': 1, 'gamma': 0.25536140493887455}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:32:43,601] Trial 6 finished with value: 0.6080628002467622 and parameters: {'n_estimators': 150, 'max_depth': 12, 'learning_rate': 0.2821139267240105, 'subsample': 0.5699239941366815, 'colsample_bytree': 0.5929973925117341, 'min_child_weight': 6, 'gamma': 0.1630538824240851}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:33:01,995] Trial 7 finished with value: 0.6089273162693261 and parameters: {'n_estimators': 234, 'max_depth': 20, 'learning_rate': 0.18191863113754334, 'subsample': 0.7104669151122349, 'colsample_bytree': 0.6252620972909657, 'min_child_weight': 2, 'gamma': 0.07371143579419281}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:33:08,671] Trial 8 finished with value: 0.6121930398050137 and parameters: {'n_estimators': 201, 'max_depth': 8, 'learning_rate': 0.219873906255665, 'subsample': 0.47151301998303513, 'colsample_bytree': 0.7187290865757359, 'min_child_weight': 9, 'gamma': 0.49891248789970066}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:33:22,101] Trial 9 finished with value: 0.6144281112459324 and parameters: {'n_estimators': 237, 'max_depth': 18, 'learning_rate': 0.19874770606215852, 'subsample': 0.73428085670826, 'colsample_bytree': 0.7844477133672275, 'min_child_weight': 1, 'gamma': 0.42863568816941616}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:33:34,585] Trial 10 finished with value: 0.6358140368232715 and parameters: {'n_estimators': 287, 'max_depth': 12, 'learning_rate': 0.03409951731440557, 'subsample': 0.6179110612622578, 'colsample_bytree': 0.42804466672377617, 'min_child_weight': 7, 'gamma': 0.01004396346324865}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:33:47,578] Trial 11 finished with value: 0.6474034739762616 and parameters: {'n_estimators': 279, 'max_depth': 12, 'learning_rate': 0.01333258466063431, 'subsample': 0.6296086724117942, 'colsample_bytree': 0.4052126308920029, 'min_child_weight': 7, 'gamma': 0.017511146123102145}. Best is trial 2 with value: 0.647567599436012.
[I 2025-01-18 19:34:00,534] Trial 12 finished with value: 0.6548378205624903 and parameters: {'n_estimators': 299, 'max_depth': 11, 'learning_rate': 0.010425293468584187, 'subsample': 0.6314521650872085, 'colsample_bytree': 0.42343193349416547, 'min_child_weight': 8, 'gamma': 0.14201151490583286}. Best is trial 12 with value: 0.6548378205624903.
[I 2025-01-18 19:34:09,749] Trial 13 finished with value: 0.6262447099766313 and parameters: {'n_estimators': 264, 'max_depth': 10, 'learning_rate': 0.07573250905780693, 'subsample': 0.5623325576672668, 'colsample_bytree': 0.479412590700757, 'min_child_weight': 10, 'gamma': 0.169370001037501}. Best is trial 12 with value: 0.6548378205624903.
[I 2025-01-18 19:34:18,172] Trial 14 finished with value: 0.6356337931743214 and parameters: {'n_estimators': 299, 'max_depth': 7, 'learning_rate': 0.06522332863485966, 'subsample': 0.6752984545299141, 'colsample_bytree': 0.4013393883792645, 'min_child_weight': 4, 'gamma': 0.17367255085036276}. Best is trial 12 with value: 0.6548378205624903.
[I 2025-01-18 19:34:27,055] Trial 15 finished with value: 0.6211763400973107 and parameters: {'n_estimators': 184, 'max_depth': 15, 'learning_rate': 0.113848255974194, 'subsample': 0.6720622356648536, 'colsample_bytree': 0.4629882033855155, 'min_child_weight': 8, 'gamma': 0.11451743188379479}. Best is trial 12 with value: 0.6548378205624903.
[I 2025-01-18 19:34:38,220] Trial 16 finished with value: 0.6370097894048461 and parameters: {'n_estimators': 264, 'max_depth': 10, 'learning_rate': 0.03675172023492762, 'subsample': 0.5369648966586881, 'colsample_bytree': 0.527593725068568, 'min_child_weight': 5, 'gamma': 0.22638184181433751}. Best is trial 12 with value: 0.6548378205624903.
[I 2025-01-18 19:34:47,717] Trial 17 finished with value: 0.657162633575237 and parameters: {'n_estimators': 202, 'max_depth': 11, 'learning_rate': 0.010143843540004211, 'subsample': 0.6325007360573056, 'colsample_bytree': 0.4417283453062639, 'min_child_weight': 8, 'gamma': 0.3322567973136216}. Best is trial 17 with value: 0.657162633575237.
[I 2025-01-18 19:34:55,943] Trial 18 finished with value: 0.658014325902223 and parameters: {'n_estimators': 183, 'max_depth': 11, 'learning_rate': 0.011183559078667799, 'subsample': 0.6211501047699268, 'colsample_bytree': 0.4463162030475032, 'min_child_weight': 10, 'gamma': 0.33855316425372695}. Best is trial 18 with value: 0.658014325902223.
[I 2025-01-18 19:35:03,662] Trial 19 finished with value: 0.6246004744254281 and parameters: {'n_estimators': 172, 'max_depth': 15, 'learning_rate': 0.10919413108800559, 'subsample': 0.5867213309230751, 'colsample_bytree': 0.5342759251790881, 'min_child_weight': 10, 'gamma': 0.34521802018551484}. Best is trial 18 with value: 0.658014325902223.
[I 2025-01-18 19:35:11,586] Trial 20 finished with value: 0.6206908544152432 and parameters: {'n_estimators': 191, 'max_depth': 13, 'learning_rate': 0.10742670485194197, 'subsample': 0.4996730965929178, 'colsample_bytree': 0.4601715329845705, 'min_child_weight': 9, 'gamma': 0.3517592735281322}. Best is trial 18 with value: 0.658014325902223.
[I 2025-01-18 19:35:20,667] Trial 21 finished with value: 0.6456605786421573 and parameters: {'n_estimators': 222, 'max_depth': 11, 'learning_rate': 0.03225051575262732, 'subsample': 0.6359399345965077, 'colsample_bytree': 0.44481516384620906, 'min_child_weight': 8, 'gamma': 0.30226872170869423}. Best is trial 18 with value: 0.658014325902223.
[I 2025-01-18 19:35:28,300] Trial 22 finished with value: 0.6596303955253021 and parameters: {'n_estimators': 164, 'max_depth': 10, 'learning_rate': 0.012810188915189686, 'subsample': 0.6384235632080573, 'colsample_bytree': 0.5112889091784111, 'min_child_weight': 8, 'gamma': 0.23449750378310394}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:35:36,619] Trial 23 finished with value: 0.6588179291879532 and parameters: {'n_estimators': 168, 'max_depth': 11, 'learning_rate': 0.010375654786835945, 'subsample': 0.5996963105624924, 'colsample_bytree': 0.5694797198236208, 'min_child_weight': 10, 'gamma': 0.23736304802450212}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:35:45,096] Trial 24 finished with value: 0.6436800394948484 and parameters: {'n_estimators': 167, 'max_depth': 10, 'learning_rate': 0.05746600198173097, 'subsample': 0.5444009760302438, 'colsample_bytree': 0.6014940558797797, 'min_child_weight': 10, 'gamma': 0.2459385497300692}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:35:53,244] Trial 25 finished with value: 0.6220527939672102 and parameters: {'n_estimators': 168, 'max_depth': 14, 'learning_rate': 0.08534220626525081, 'subsample': 0.5970694362210528, 'colsample_bytree': 0.5650648604719645, 'min_child_weight': 10, 'gamma': 0.3944300287853494}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:35:59,630] Trial 26 finished with value: 0.6110104737306294 and parameters: {'n_estimators': 178, 'max_depth': 9, 'learning_rate': 0.2458035772161027, 'subsample': 0.7132611647533722, 'colsample_bytree': 0.519232340180853, 'min_child_weight': 9, 'gamma': 0.2084919601349315}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:36:06,564] Trial 27 finished with value: 0.6462080070010471 and parameters: {'n_estimators': 150, 'max_depth': 13, 'learning_rate': 0.038377623211519885, 'subsample': 0.5290660260740351, 'colsample_bytree': 0.5531743121854444, 'min_child_weight': 10, 'gamma': 0.27791546110677007}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:36:13,525] Trial 28 finished with value: 0.6299642117838988 and parameters: {'n_estimators': 191, 'max_depth': 9, 'learning_rate': 0.09568333809749294, 'subsample': 0.5982936666720918, 'colsample_bytree': 0.6380799870194307, 'min_child_weight': 9, 'gamma': 0.20744735298614003}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:36:20,554] Trial 29 finished with value: 0.6221803032111382 and parameters: {'n_estimators': 164, 'max_depth': 11, 'learning_rate': 0.1400720379314909, 'subsample': 0.6470448422386919, 'colsample_bytree': 0.5746000912477464, 'min_child_weight': 9, 'gamma': 0.49136776445440067}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:36:27,463] Trial 30 finished with value: 0.6205759952158693 and parameters: {'n_estimators': 160, 'max_depth': 16, 'learning_rate': 0.13108859205820916, 'subsample': 0.4400286838547901, 'colsample_bytree': 0.5037732292696583, 'min_child_weight': 9, 'gamma': 0.08210208202176572}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:36:36,790] Trial 31 finished with value: 0.6551435615843753 and parameters: {'n_estimators': 202, 'max_depth': 11, 'learning_rate': 0.012309141188473852, 'subsample': 0.608433190328262, 'colsample_bytree': 0.4724819395465927, 'min_child_weight': 8, 'gamma': 0.32043336775205744}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:36:45,078] Trial 32 finished with value: 0.6499099157911565 and parameters: {'n_estimators': 192, 'max_depth': 10, 'learning_rate': 0.024799888996611872, 'subsample': 0.5781845873036712, 'colsample_bytree': 0.44314089817474617, 'min_child_weight': 8, 'gamma': 0.42041818082846305}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:36:54,271] Trial 33 finished with value: 0.6353554202867251 and parameters: {'n_estimators': 176, 'max_depth': 12, 'learning_rate': 0.050504552863389174, 'subsample': 0.6549534143974378, 'colsample_bytree': 0.4949467838288881, 'min_child_weight': 7, 'gamma': 0.3309508716763133}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:37:02,216] Trial 34 finished with value: 0.6402429200515851 and parameters: {'n_estimators': 210, 'max_depth': 9, 'learning_rate': 0.052383367695330155, 'subsample': 0.40020676597220506, 'colsample_bytree': 0.5427711249198328, 'min_child_weight': 6, 'gamma': 0.48321726040846114}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:37:12,721] Trial 35 finished with value: 0.6453456124064858 and parameters: {'n_estimators': 160, 'max_depth': 13, 'learning_rate': 0.02624460937226616, 'subsample': 0.7934478475093738, 'colsample_bytree': 0.4441956560115632, 'min_child_weight': 5, 'gamma': 0.37488048641439803}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:37:19,137] Trial 36 finished with value: 0.6400132828430911 and parameters: {'n_estimators': 184, 'max_depth': 8, 'learning_rate': 0.0720569621822235, 'subsample': 0.6911722220936646, 'colsample_bytree': 0.5063580172865485, 'min_child_weight': 10, 'gamma': 0.5854817678810369}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:37:28,286] Trial 37 finished with value: 0.6337480133227913 and parameters: {'n_estimators': 200, 'max_depth': 11, 'learning_rate': 0.047736131579847685, 'subsample': 0.6147448373571665, 'colsample_bytree': 0.47867852668489624, 'min_child_weight': 7, 'gamma': 0.26912821191017056}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:37:38,269] Trial 38 finished with value: 0.6496298389917065 and parameters: {'n_estimators': 216, 'max_depth': 10, 'learning_rate': 0.02031459970547987, 'subsample': 0.698916884269267, 'colsample_bytree': 0.5873086437540206, 'min_child_weight': 8, 'gamma': 0.3023576834113422}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:37:44,130] Trial 39 finished with value: 0.6507602926665456 and parameters: {'n_estimators': 178, 'max_depth': 7, 'learning_rate': 0.04191490384816794, 'subsample': 0.6522355395137193, 'colsample_bytree': 0.6137992494703655, 'min_child_weight': 9, 'gamma': 0.20733614707089917}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:37:50,324] Trial 40 finished with value: 0.6351584984834571 and parameters: {'n_estimators': 154, 'max_depth': 8, 'learning_rate': 0.064643474287417, 'subsample': 0.5662058219555594, 'colsample_bytree': 0.6691509711912067, 'min_child_weight': 6, 'gamma': 0.24225310664412203}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:37:59,866] Trial 41 finished with value: 0.6480678399487513 and parameters: {'n_estimators': 202, 'max_depth': 11, 'learning_rate': 0.02186958378924216, 'subsample': 0.6101475483014508, 'colsample_bytree': 0.4827878374434271, 'min_child_weight': 8, 'gamma': 0.32694559034628073}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:38:10,498] Trial 42 finished with value: 0.6551729151070191 and parameters: {'n_estimators': 210, 'max_depth': 12, 'learning_rate': 0.013407534274054663, 'subsample': 0.6598887028069295, 'colsample_bytree': 0.46522732599098604, 'min_child_weight': 9, 'gamma': 0.30585243587461997}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:38:20,813] Trial 43 finished with value: 0.6531354885256991 and parameters: {'n_estimators': 214, 'max_depth': 12, 'learning_rate': 0.011037227189061149, 'subsample': 0.658048814199485, 'colsample_bytree': 0.423102855088806, 'min_child_weight': 10, 'gamma': 0.3823331667689471}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:38:30,800] Trial 44 finished with value: 0.6125659634883801 and parameters: {'n_estimators': 224, 'max_depth': 12, 'learning_rate': 0.16857738642027797, 'subsample': 0.7149862565946347, 'colsample_bytree': 0.518923149342318, 'min_child_weight': 9, 'gamma': 0.2818273885803983}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:38:41,359] Trial 45 finished with value: 0.6417637123229699 and parameters: {'n_estimators': 196, 'max_depth': 13, 'learning_rate': 0.029400155069413347, 'subsample': 0.7434453245285778, 'colsample_bytree': 0.45263228215695045, 'min_child_weight': 7, 'gamma': 0.4558141882292956}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:38:48,576] Trial 46 finished with value: 0.6400620095213408 and parameters: {'n_estimators': 185, 'max_depth': 9, 'learning_rate': 0.04551022734198851, 'subsample': 0.6275563793196516, 'colsample_bytree': 0.4241600126000904, 'min_child_weight': 10, 'gamma': 0.3604630878578808}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:39:01,261] Trial 47 finished with value: 0.6441188346035498 and parameters: {'n_estimators': 243, 'max_depth': 14, 'learning_rate': 0.02465317129910178, 'subsample': 0.553885224386607, 'colsample_bytree': 0.7716747988698107, 'min_child_weight': 9, 'gamma': 0.39842771314276293}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:39:14,265] Trial 48 finished with value: 0.6066393642978467 and parameters: {'n_estimators': 232, 'max_depth': 19, 'learning_rate': 0.2673526053087848, 'subsample': 0.6773200626992019, 'colsample_bytree': 0.552246004751419, 'min_child_weight': 4, 'gamma': 0.3034759047558681}. Best is trial 22 with value: 0.6596303955253021.
[I 2025-01-18 19:39:22,579] Trial 49 finished with value: 0.6267634754304833 and parameters: {'n_estimators': 207, 'max_depth': 10, 'learning_rate': 0.0870933111226585, 'subsample': 0.584188224228918, 'colsample_bytree': 0.4941633149430661, 'min_child_weight': 8, 'gamma': 0.25989206860777125}. Best is trial 22 with value: 0.6596303955253021.
Best Parameters (Temporal Set): {'n_estimators': 164, 'max_depth': 10, 'learning_rate': 0.012810188915189686, 'subsample': 0.6384235632080573, 'colsample_bytree': 0.5112889091784111, 'min_child_weight': 8, 'gamma': 0.23449750378310394}
Best F1 Score (Temporal Set): 0.6596303955253021


Validation Metrics (Temporal Set):
Accuracy: 0.6988670244484197
Precision: 0.6197874080130826
Recall: 0.5817344589409056
F1-Score: 0.6001583531274742
AUC-ROC: 0.6775078925616279
Confusion Matrix (Temporal Set):
[[1586  465]
 [ 545  758]]

Test Metrics (Random Set - XGBoost):
Accuracy: 0.6568276684555754
Precision: 0.6084848484848485
Recall: 0.6653412856196156
F1-Score: 0.6356441911997468
AUC-ROC: 0.7211992722975606
Confusion Matrix:
[[1199  646]
 [ 505 1004]]

Test Metrics (Temporal Set - XGBoost):
Accuracy: 0.7158616577221228
Precision: 0.687382297551789
Recall: 0.540340488527017
F1-Score: 0.6050559469539991
AUC-ROC: 0.7769330460268147
Confusion Matrix:
[[1671  332]
 [ 621  730]]
 """