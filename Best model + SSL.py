"""THIS IS THE LAST NOTEBOOK/EXPERIMENT
Using the best model in my SSL experiment to see if i can improve metrics, i am expecially interested in recall improvement. 

The best model was the XGBoost on a temporal split, which gave the following results on the test set:
Test Metrics (Temporal Set - XGBoost):
Accuracy: 0.7152653548002386
Precision: 0.6892925430210325
Recall: 0.533678756476684
F1-Score: 0.6015853149770546
AUC-ROC: 0.7802864541086224
Confusion Matrix:
[[1678  325]
 [ 630  721]]

 After the experiment i got these results:
Test Metrics (SSL with XGBoost):
Accuracy: 0.6961836612999404
Precision: 0.6263318112633182
Recall: 0.609178386380459
F1-Score: 0.6176360225140713
AUC-ROC: 0.7664999539920319
Confusion Matrix:
[[1512  491]
 [ 528  823]]


 These were the hyperparameters for the model:
 Fitting RandomizedSearchCV for Temporal Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'subsample': 0.4, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.4}
Best F1 Score (Temporal Set): 0.6594620380425328

"""

# Importing necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.semi_supervised import LabelPropagation
import graphviz

# Load the balanced temporal training and validation datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')
X_test_temp = pd.read_csv('X_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv')


#load the unlabeled dataset
unlabeled_data_encoded_df = pd.read_csv('unlabeled_data_encoded_df.csv')


# Initialize the XGBoost classifier with randomstate 777
xgb_model = XGBClassifier(random_state=777, eval_metric='logloss') 

# Custom function to create stratified time-series splits
def stratified_time_series_split(X, y, n_splits=5):
    # Create a list of indices to hold the splits
    indices = []
    
    # Initialize the StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)

    # Generate the indices for stratified time-series split
    for train_index, val_index in stratified_kfold.split(X, y):
        # Ensure the maintainance of the temporal order
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


# Hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [150, 170, 200, 250, 270],
    'max_depth': [ 7, 10, 12, 15, 20],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.4, 0.5, 0.6, 0.7],
    'colsample_bytree': [0.4, 0.5, 0.6, 0.7],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.4, 0.5, 0.6]
}


# RandomizedSearchCV for Temporal Set
print("\nFitting RandomizedSearchCV for Temporal Set...")
random_search_temp = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter= 50,
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


# Combine training and validation datasets for Temporal Set
X_train_val_temp = pd.concat([X_temp_balanced, x_val_temp], axis=0)
y_train_val_temp = pd.concat([y_temp_balanced, y_val_temp], axis=0)

# Fit final XGBoost model on the combined dataset for the Temporal Set
final_model_temp_xgb = XGBClassifier(**random_search_temp.best_params_, random_state=777, eval_metric='logloss')
final_model_temp_xgb.fit(X_train_val_temp, y_train_val_temp)


#PERMUTATION IMPORTANCE

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


#SSL EXPERIMENT

# Initialize the LabelPropagation model using a KNN model and 5 neighbors
lp_model = LabelPropagation(kernel='knn', n_neighbors=5)

# Combine labeled and unlabeled data
X_combined = pd.concat([X_train_val_temp, unlabeled_data_encoded_df], axis=0)
y_combined = pd.concat([y_train_val_temp['Target_binary'], pd.Series([-1] * len(unlabeled_data_encoded_df))], axis=0)# -1 in this case represents unlabeled data

# Fit the Label Propagation model
lp_model.fit(X_combined, y_combined)

# Get the pseudo-labels for the unlabeled df
pseudo_labels = lp_model.transduction_[len(X_train_val_temp):]

# Combine the labeled data with pseudo-labeled data
X_train_pseudo = pd.concat([X_train_val_temp, unlabeled_data_encoded_df], axis=0)
y_train_pseudo = pd.concat([y_train_val_temp['Target_binary'], pd.Series(pseudo_labels)], axis=0)  


# Train a new XGBoost model on the combined data with pseudo-labels
final_model_ssl_xgb = XGBClassifier(**random_search_temp.best_params_,scale_pos_weight=1.5, random_state=777, eval_metric='logloss') #change the weight to 1.4 to attempt to imrpove recall  while maintaining other parameters
final_model_ssl_xgb.fit(X_train_pseudo, y_train_pseudo)

# Test Set Evaluation for the SSL model
test_ssl_predictions_xgb = final_model_ssl_xgb.predict(X_test_temp)
test_ssl_f1_xgb = f1_score(y_test_temp, test_ssl_predictions_xgb)
test_ssl_accuracy_xgb = accuracy_score(y_test_temp, test_ssl_predictions_xgb)
test_ssl_precision_xgb = precision_score(y_test_temp, test_ssl_predictions_xgb)
test_ssl_recall_xgb = recall_score(y_test_temp, test_ssl_predictions_xgb)
test_ssl_roc_auc_xgb = roc_auc_score(y_test_temp, final_model_ssl_xgb.predict_proba(X_test_temp)[:, 1])
test_ssl_conf_matrix_xgb = confusion_matrix(y_test_temp, test_ssl_predictions_xgb)

print("\nTest Metrics (SSL with XGBoost):")
print(f"Accuracy: {test_ssl_accuracy_xgb}")
print(f"Precision: {test_ssl_precision_xgb}")
print(f"Recall: {test_ssl_recall_xgb}")
print(f"F1-Score: {test_ssl_f1_xgb}")
print(f"AUC-ROC: {test_ssl_roc_auc_xgb}")
print(f"Confusion Matrix:\n{test_ssl_conf_matrix_xgb}")



