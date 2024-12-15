# Importing necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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

 
 """