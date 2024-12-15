# Importing libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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
    y_train, y_val = y_temp_balanced[train_index], y_temp_balanced[val_index]
    
    # Fit the model and calculate F1 score on the validation set
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_val)
    
    # Calculate the F1 score for this fold
    f1_score_temp = f1_score(y_val, y_pred)
    temporal_cv_scores.append(f1_score_temp)

# Print the results
print("Temporal Set - Stratified Time Series Cross-Validation F1 Scores:", temporal_cv_scores)
print("Temporal Set - Mean F1 Score:", np.mean(temporal_cv_scores))


# Hyperparameter tuning using RandomizedSearchCV focused on F1 score
param_dist_rf = {
    'n_estimators': [70, 80, 130, 150, 200],
    'max_depth': [None, 20, 25, 27, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}

# RandomizedSearchCV for Random Set
random_search_rand_rf = RandomizedSearchCV(
    estimator=model_rf,
    param_distributions=param_dist_rf,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=777
)
random_search_rand_rf.fit(X_rand_balanced, y_rand_balanced)
print("Best Parameters (Random Set):", random_search_rand_rf.best_params_)
print("Best F1 Score (Random Set):", random_search_rand_rf.best_score_)

# RandomizedSearchCV for Temporal Set
random_search_temp_rf = RandomizedSearchCV(
    estimator=model_rf,
    param_distributions=param_dist_rf,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=777
)
random_search_temp_rf.fit(X_temp_balanced, y_temp_balanced)
print("Best Parameters (Temporal Set):", random_search_temp_rf.best_params_)
print("Best F1 Score (Temporal Set):", random_search_temp_rf.best_score_)

# Validation Metrics for Random Set
best_model_rand_rf = random_search_rand_rf.best_estimator_
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
best_model_temp_rf = random_search_temp_rf.best_estimator_
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

# Refit the best model for the Random Set
final_model_rand_rf = RandomForestClassifier(**random_search_rand_rf.best_params_, random_state=777, n_jobs=-1)
final_model_rand_rf.fit(X_train_val_rand, y_train_val_rand)

# Refit the best model for the Temporal Set
final_model_temp_rf = RandomForestClassifier(**random_search_temp_rf.best_params_, random_state=777, n_jobs=-1)
final_model_temp_rf.fit(X_train_val_temp, y_train_val_temp)





#permutation importance
# Permutation Importance for Random Set
perm_importance_rand_rf = permutation_importance(final_model_rand_rf, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
importance_rand_df_rf = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Importance': perm_importance_rand_rf.importances_mean
})
importance_rand_df_rf = importance_rand_df_rf[(importance_rand_df_rf['Importance'] > 0.001) | (importance_rand_df_rf['Importance'] < -0.001)] #cutt off to avoid clutter in the plot
importance_rand_df_rf.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_rand_df_rf)
plt.title('Permutation Importance (Random Set)')
plt.show()

# Permutation Importance for Temporal Set
perm_importance_temp_rf = permutation_importance(final_model_temp_rf, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
importance_temp_df_rf = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp_rf.importances_mean
})
importance_temp_df_rf = importance_temp_df_rf[(importance_temp_df_rf['Importance'] > 0.001) | (importance_temp_df_rf['Importance'] < -0.001)]
importance_temp_df_rf.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for Temporal Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_temp_df_rf)
plt.title('Permutation Importance (Temporal Set)')
plt.show()


# final Test Set Evaluation for Random Set
final_model_rand_rf = random_search_rand_rf.best_estimator_
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
final_model_temp_rf = random_search_temp_rf.best_estimator_
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

"""
