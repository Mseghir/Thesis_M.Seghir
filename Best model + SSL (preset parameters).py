from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.semi_supervised import LabelPropagation
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the balanced temporal training and validation datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')
X_test_temp = pd.read_csv('X_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv')

# Load the unlabeled dataset
unlabeled_data_encoded_df = pd.read_csv('unlabeled_data_encoded_df.csv')

# Preset parameters for XGBoost based on the best performing model
best_params_temp = {
    'n_estimators': 164,
    'max_depth': 10,
    'learning_rate': 0.012810188915189686,
    'subsample': 0.6384235632080573,
    'colsample_bytree': 0.5112889091784111,
    'min_child_weight': 8,
    'gamma': 0.23449750378310394,
}

# Combine training and validation datasets for Temporal Set
X_train_val_temp = pd.concat([X_temp_balanced, x_val_temp], axis=0)
y_train_val_temp = pd.concat([y_temp_balanced, y_val_temp], axis=0)

# Train the final XGBoost model with the preset parameters
final_model_temp_xgb = XGBClassifier(**best_params_temp, random_state=777, eval_metric='logloss')
final_model_temp_xgb.fit(X_train_val_temp, y_train_val_temp)

# Permutation Importance for the Temporal Set
perm_importance_temp_xgb = permutation_importance(final_model_temp_xgb, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
importance_temp_df_xgb = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp_xgb.importances_mean
})

# Plot Permutation Importance for the Temporal Set
importance_temp_df_xgb = importance_temp_df_xgb[(importance_temp_df_xgb['Importance'] > 0.001) | (importance_temp_df_xgb['Importance'] < -0.001)]
importance_temp_df_xgb.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for the Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_temp_df_xgb)
plt.title('Permutation Importance XGBoost (temporal split)  + SSL')
plt.show()

# Semi-Supervised Learning Experiment with SSL
lp_model = LabelPropagation(kernel='knn', n_neighbors=5)

# Combine labeled and unlabeled data
X_combined = pd.concat([X_train_val_temp, unlabeled_data_encoded_df], axis=0)
y_combined = pd.concat([y_train_val_temp['Target_binary'], pd.Series([-1] * len(unlabeled_data_encoded_df))], axis=0)  # -1 in this case represents unlabeled data

# Fit the Label Propagation model
lp_model.fit(X_combined, y_combined)

# Get the pseudo-labels for the unlabeled df
pseudo_labels = lp_model.transduction_[len(X_train_val_temp):]

# Combine the labeled data with pseudo-labeled data
X_train_pseudo = pd.concat([X_train_val_temp, unlabeled_data_encoded_df], axis=0)
y_train_pseudo = pd.concat([y_train_val_temp['Target_binary'], pd.Series(pseudo_labels)], axis=0)

# Train a new XGBoost model on the combined data with pseudo-labels
final_model_ssl_xgb = XGBClassifier(**best_params_temp, scale_pos_weight=1.5, random_state=777, eval_metric='logloss')
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

# Apply PCA to reduce dimensions to 2D for visualization
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Perform PCA transformation on the test data
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_temp)

# Ensure y_test_temp is a 1D array (flatten if necessary)
if isinstance(y_test_temp, pd.DataFrame):
    y_test_temp = y_test_temp.values.flatten()  # Flatten to a 1D array

# Plot 1: True Class Labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test_temp, palette='coolwarm', alpha=0.7)
plt.title("PCA Visualization of Test Data (True Class Labels)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="True Class")
plt.show()

# Plot 2: SSL Pseudo-Labels
pseudo_labels_test = final_model_ssl_xgb.predict(X_test_temp)  # Get predictions from SSL model

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=pseudo_labels_test, palette='coolwarm', alpha=0.7)
plt.title("PCA Visualization of Test Data (SSL Pseudo-Labels)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Pseudo-Labels")
plt.show()

# Apply t-SNE to reduce dimensions to 2D for visualization
tsne = TSNE(n_components=2, random_state=777)
X_test_tsne = tsne.fit_transform(X_test_temp)

# Plot 3: True Class Labels (t-SNE)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_tsne[:, 0], y=X_test_tsne[:, 1], hue=y_test_temp, palette='coolwarm', alpha=0.7)
plt.title("t-SNE Visualization of Test Data (True Class Labels)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="True Class")
plt.show()

# Plot 4: SSL Pseudo-Labels (t-SNE)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_tsne[:, 0], y=X_test_tsne[:, 1], hue=pseudo_labels_test, palette='coolwarm', alpha=0.7)
plt.title("t-SNE Visualization of Test Data (SSL Pseudo-Labels)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Pseudo-Labels")
plt.show()


"""Test Metrics (SSL with XGBoost):
Accuracy: 0.6911150864639237
Precision: 0.6275303643724697
Recall: 0.5736491487786824
F1-Score: 0.5993812838360403
AUC-ROC: 0.7615177899324219
Confusion Matrix:
[[1543  460]
 [ 576  775]]"""