# AFTER CLEANING RUN THIS FILE
#Preprocessing the features so that the decision tree will be able to correctly process the data
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from matplotlib_venn import venn2
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Reading cleaned dataset

df =pd.read_csv('cleaned_data.csv')
print(df.head())
print(df.info())
print(df.describe())


# Data split into labeled and unlabeled data
labeled_data = df[df['Target'].notna()].copy()
unlabeled_data = df[df['Target'].isna()].copy()

# Save the labeled and unlabeled data to separate CSV files
labeled_data.to_csv('labeled_data.csv', index=False)
unlabeled_data.to_csv('unlabeled_data.csv', index=False)

# Checking for differences between missing data % in labelled and unlabelled data to check if i should do more missing data imputation
def display_missing_values(df, max_columns=None, max_rows=None):
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)

    missing_value_percentages = df.isnull().mean() * 100
    missing_value_percentages = missing_value_percentages.sort_values(ascending=False)
    
    print(missing_value_percentages)

display_missing_values(df,max_columns=None, max_rows=None)
print(f"Missing data % in labeled data:")
display_missing_values(labeled_data, max_columns=None, max_rows=None)
print(f"Missing data % in unlabeled data:")
display_missing_values(unlabeled_data, max_columns=None, max_rows=None)


# Dropping rows with missing values in unlabeled_data, EXCLUDING the 'Target' column (will need it for feature engineering to keep the amount of columns the same.. even if the result is empty columns)
unlabeled_data = unlabeled_data[unlabeled_data.drop(columns=['Target']).notna().all(axis=1)]


#Doing some feature transformation and dropping columns
# Transform "Target" into a binary column in labeled data
labeled_data['Target_binary'] = np.where(labeled_data['Contract_duur'] > 3, 1, 0)

#Dropping rows with missing values
labeled_data = labeled_data.dropna()

# Function to plot correlation matrix, excluding categorical data that does not correlate numerically
def plot_correlation_matrix(df, title):
    plt.figure(figsize=(24, 20))
    
    # Selecting only numeric columns for correlation calculation in the plot
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    # Focus only on the correlation with target variable
    if 'Target_binary' in corr_matrix.columns:
        target_corr = corr_matrix[['Target_binary']].sort_values(by='Target_binary', ascending=False)
        
        # Plot the heatmap
        sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(title)
        plt.show()
    else:
        print("'Target_binary' not found in the correlation matrix.")



print("Overal correlation")
plot_correlation_matrix(labeled_data, "Correlation with Binary Target")

#Removing letters from postal code, this loses some of its granularity but it's still more precise then a city name, also more precise than a region
labeled_data['Postcode_cijfers'] = labeled_data['Postcode'].str.replace(r'[A-Za-z]', '', regex=True)
labeled_data = labeled_data.drop(columns=["Postcode"])

unlabeled_data['Postcode_cijfers'] = unlabeled_data['Postcode'].str.replace(r'[A-Za-z]', '', regex=True)
unlabeled_data = unlabeled_data.drop(columns=["Postcode"])


# Convert columns to datetime using .loc to avoid SettingWithCopyWarning
labeled_data.loc[:, 'Construction'] = pd.to_datetime(labeled_data['Year of construction'])
labeled_data.loc[:, 'Demolition'] = pd.to_datetime(labeled_data['Year of demolition'])
labeled_data.loc[:, 'VABI_finished'] = pd.to_datetime(labeled_data['Amfelddatum_VABI'])
labeled_data.loc[:, 'Contract_starting'] = pd.to_datetime(labeled_data['Ingangsdatum_contract'])

unlabeled_data.loc[:, 'Construction'] = pd.to_datetime(unlabeled_data['Year of construction'])
unlabeled_data.loc[:, 'Demolition'] = pd.to_datetime(unlabeled_data['Year of demolition'])
unlabeled_data.loc[:, 'VABI_finished'] = pd.to_datetime(unlabeled_data['Amfelddatum_VABI'])
unlabeled_data.loc[:, 'Contract_starting'] = pd.to_datetime(unlabeled_data['Ingangsdatum_contract'])

# Extract year from all datetime columns using .loc to set new columns
for col in labeled_data.select_dtypes(include=['datetime']).columns:
    labeled_data.loc[:, f'{col}_year'] = labeled_data[col].dt.year

for col in unlabeled_data.select_dtypes(include=['datetime']).columns:
    unlabeled_data.loc[:, f'{col}_year'] = unlabeled_data[col].dt.year

# Drop the original columns with .loc
labeled_data.drop(columns=['Year of construction', 'Year of demolition', 'Amfelddatum_VABI', 'Ingangsdatum_contract'], inplace=True)

unlabeled_data.drop(columns=['Year of construction', 'Year of demolition', 'Amfelddatum_VABI', 'Ingangsdatum_contract'], inplace=True)


#FEATURES TO BE INCLUDED BEFORE SPLIT
labeled_data['Totaal_Aantal_Historische_Huurders'] = (
    labeled_data.groupby('VIBDRO_Huurobject_id')['Contractnummer'].transform('nunique')
)

labeled_data = labeled_data.sort_values(by=['VIBDRO_Huurobject_id', 'Contract_starting'])

labeled_data['Aantal_Historische_Huurders_Vanaf_contractdatum'] = (
    labeled_data.groupby('VIBDRO_Huurobject_id').cumcount() + 1
)

labeled_data['Contract_duur_vorige_huurder'] = (
    labeled_data.groupby('VIBDRO_Huurobject_id')['Target'].shift(1)
).fillna('n.v.t.')

encoding_map = {
    'n.v.t.': -1,
    '<=3': 0,
    '>3': 1
}

# Apply the encoding 
labeled_data['Contract_duur_vorige_huurder_encoded'] = labeled_data['Contract_duur_vorige_huurder'].map(encoding_map)

# Feature: Total historical tenants with Target <= 3
labeled_data['Totaal_aantal_historisch_huurders_kort'] = (
    labeled_data[labeled_data['Target'] == '<= 3' ]
    .groupby('VIBDRO_Huurobject_id')['Contractnummer']
    .transform('nunique')
)

# Feature: Total historical tenants with Target > 3
labeled_data['Totaal_aantal_historisch_huurders_lang'] = (
    labeled_data[labeled_data['Target'] == '> 3' ]
    .groupby('VIBDRO_Huurobject_id')['Contractnummer']
    .transform('nunique')
)

# Fill NaN values with 0 for cases where no tenants match the condition
labeled_data['Totaal_aantal_historisch_huurders_kort'] = labeled_data['Totaal_aantal_historisch_huurders_kort'].fillna(0)
labeled_data['Totaal_aantal_historisch_huurders_lang'] = labeled_data['Totaal_aantal_historisch_huurders_lang'].fillna(0)

# Proportion of short contracts (Target <= 3)
labeled_data['Proportie_historisch_huurders_kort'] = (
    labeled_data['Totaal_aantal_historisch_huurders_kort'] / labeled_data['Totaal_Aantal_Historische_Huurders']
)

# Proportion of long contracts (Target > 3)
labeled_data['Proportie_historisch_huurders_lang'] = (
    labeled_data['Totaal_aantal_historisch_huurders_lang'] / labeled_data['Totaal_Aantal_Historische_Huurders']
)

# Handle division by zero: if there are no historical contracts, set proportions to 0
labeled_data['Proportie_historisch_huurders_kort'] = labeled_data['Proportie_historisch_huurders_kort'].fillna(0)
labeled_data['Proportie_historisch_huurders_lang'] = labeled_data['Proportie_historisch_huurders_lang'].fillna(0)

labeled_data['Kort_Lang_Ratio'] = (
    labeled_data['Totaal_aantal_historisch_huurders_kort'] / 
    (labeled_data['Totaal_aantal_historisch_huurders_lang'] + 1e-9)  # Adding this very small value to avoid division by zero
)

labeled_data['Lang_Kort_Ratio'] = (
    labeled_data['Totaal_aantal_historisch_huurders_lang'] / 
    (labeled_data['Totaal_aantal_historisch_huurders_kort'] + 1e-9) #doing the same here
)

# Huurdersomloopsnelheid (Turnover Rate)
labeled_data['Huurdersomloopsnelheid'] = (
    labeled_data['Totaal_Aantal_Historische_Huurders'] /
    labeled_data['Huis_leeftijd']
).replace([float('inf'), -float('inf')], 0).fillna(0)


#doing the same for unlabeled data
unlabeled_data['Totaal_Aantal_Historische_Huurders'] = (
    unlabeled_data.groupby('VIBDRO_Huurobject_id')['Contractnummer'].transform('nunique')
)

unlabeled_data = unlabeled_data.sort_values(by=['VIBDRO_Huurobject_id', 'Contract_starting'])

unlabeled_data['Aantal_Historische_Huurders_Vanaf_contractdatum'] = (
    unlabeled_data.groupby('VIBDRO_Huurobject_id').cumcount() + 1
)

unlabeled_data['Contract_duur_vorige_huurder'] = (
    unlabeled_data.groupby('VIBDRO_Huurobject_id')['Target'].shift(1)
).fillna('n.v.t.')

encoding_map = {
    'n.v.t.': -1,
    '<=3': 0,
    '>3': 1
}

# Apply the encoding 
unlabeled_data['Contract_duur_vorige_huurder_encoded'] = unlabeled_data['Contract_duur_vorige_huurder'].map(encoding_map)

# Feature: Total historical tenants with Target <= 3
unlabeled_data['Totaal_aantal_historisch_huurders_kort'] = (
    unlabeled_data[unlabeled_data['Target'] == '<= 3']
    .groupby('VIBDRO_Huurobject_id')['Contractnummer']
    .transform('nunique')
)

# Feature: Total historical tenants with Target > 3
unlabeled_data['Totaal_aantal_historisch_huurders_lang'] = (
    unlabeled_data[unlabeled_data['Target'] == '> 3']
    .groupby('VIBDRO_Huurobject_id')['Contractnummer']
    .transform('nunique')
)

# Fill NaN values with 0 for cases where no tenants match the condition
unlabeled_data['Totaal_aantal_historisch_huurders_kort'] = unlabeled_data['Totaal_aantal_historisch_huurders_kort'].fillna(0)
unlabeled_data['Totaal_aantal_historisch_huurders_lang'] = unlabeled_data['Totaal_aantal_historisch_huurders_lang'].fillna(0)

# Proportion of short contracts (Target <= 3)
unlabeled_data['Proportie_historisch_huurders_kort'] = (
    unlabeled_data['Totaal_aantal_historisch_huurders_kort'] / unlabeled_data['Totaal_Aantal_Historische_Huurders']
)

# Proportion of long contracts (Target > 3)
unlabeled_data['Proportie_historisch_huurders_lang'] = (
    unlabeled_data['Totaal_aantal_historisch_huurders_lang'] / unlabeled_data['Totaal_Aantal_Historische_Huurders']
)

# Handle division by zero: if there are no historical contracts, set proportions to 0
unlabeled_data['Proportie_historisch_huurders_kort'] = unlabeled_data['Proportie_historisch_huurders_kort'].fillna(0)
unlabeled_data['Proportie_historisch_huurders_lang'] = unlabeled_data['Proportie_historisch_huurders_lang'].fillna(0)

unlabeled_data['Kort_Lang_Ratio'] = (
    unlabeled_data['Totaal_aantal_historisch_huurders_kort'] / 
    (unlabeled_data['Totaal_aantal_historisch_huurders_lang'] + 1e-9)  # Adding this very small value to avoid division by zero
)

unlabeled_data['Lang_Kort_Ratio'] = (
    unlabeled_data['Totaal_aantal_historisch_huurders_lang'] / 
    (unlabeled_data['Totaal_aantal_historisch_huurders_kort'] + 1e-9) #doing the same here
)

# Huurdersomloopsnelheid (Turnover Rate)
unlabeled_data['Huurdersomloopsnelheid'] = (
    unlabeled_data['Totaal_Aantal_Historische_Huurders'] /
    unlabeled_data['Huis_leeftijd']
).replace([float('inf'), -float('inf')], 0).fillna(0)

# TEMPORAL SPLIT AND RANDOM SPLIT
#I am splitting the data before feature engineering because i want to avoid data leakage, so that new data doesnt influence old data in the temporal split


# Perform a random split to determine the training set size for a 60/20/20 split
train_data_rand, temp_data_rand = train_test_split(labeled_data, test_size=0.4, random_state=777)  # 40% for validation + test
validation_data_rand, test_data_rand = train_test_split(temp_data_rand, test_size=0.5, random_state=777)  # Split the 40% into 20% validation and 20% test

# Get the size of the training set from the random split
train_size = train_data_rand.shape[0]

# Sort the labeled data by 'House age' in descending order (oldest houses first)
labeled_data_sorted = labeled_data.sort_values(by='Huis_leeftijd', ascending=False)

# Calculate sizes for a 60/20/20 split
train_size = int(0.6 * labeled_data_sorted.shape[0])  
temp_size = labeled_data_sorted.shape[0] - train_size 

# Select the top 'train_size' so both sets are equally large
train_data_temp = labeled_data_sorted.iloc[:train_size]

# The remaining rows will be split into validation and test set
temp_data_temp = labeled_data_sorted.iloc[train_size:]

# Now split the remaining data into validation and test sets (both 20%)
validation_data_temp, test_data_temp = train_test_split(temp_data_temp, test_size=0.5, random_state=777)  


# Display the sizes to verify the split
print("Training set size (random):", train_data_rand.shape[0])
print("Validation set size (random):", validation_data_rand.shape[0])
print("Testing set size (random):", test_data_rand.shape[0])

# Check the sizes of the splits
print("Training set size (temporal):", train_data_temp.shape[0])
print("Validation set size (temporal):", validation_data_temp.shape[0])
print("Testing set size (temporal):", test_data_temp.shape[0])

print("Total size:", labeled_data.shape[0])

#CHECK FOR TARGET CLASS BALANCE IN BOTH SETS:
#Plot class balance for both classes
def check_class_balance(data, target_variable, title):
    class_counts = data[target_variable].value_counts()
    print(f"Class distribution for {title}:")
    print(class_counts)

    # Plotting class balance
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values)
    plt.title(f'Class Balance for {title}')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

# Checking class balance for the temporal training split
check_class_balance(train_data_temp, 'Target_binary', 'Temporal Training Split')

# Checking class balance for the validation set (temporal)
check_class_balance(validation_data_temp, 'Target_binary', 'Temporal Validation Split')

# Checking class balance for the random training split
check_class_balance(train_data_rand, 'Target_binary', 'Random Training Split')

# Checking class balance for the validation set (random)
check_class_balance(validation_data_rand, 'Target_binary', 'Random Validation Split')

#There is a class imbalance in all instances where 1 is overrepresentend and 0 is underrepresented.The class imbalance is quite bad so i need SMOTE. Im doing SMOTE after feature engineering to 
#Make sure classes are balanced in training data 
#After running only data after 2010 the class imbalance is gone. See the following results after removing data from before and after dropping data from 2010:


"""
CLASS BALANCE BEFORE IN ALL CONTRACTS:

Training set size (random): 18769
Validation set size (random): 6256
Testing set size (random): 6257

Training set size (temporal): 18769
Validation set size (temporal): 6256
Testing set size (temporal): 6257
Total size: 31282

Class distribution for Temporal Training Split:
1    13630
0     5139

Name: Target_binary, dtype: int64
Class distribution for Temporal Validation Split:
0    3249
1    3007

Name: Target_binary, dtype: int64
Class distribution for Random Training Split:
1    11801
0     6968

Name: Target_binary, dtype: int64
Class distribution for Random Validation Split:
1    3963
0    2293

CLASS BALANCE AFTER REMOVING CONTRACTS FROM BEFORE 2010:
Training set size (random): 12151
Validation set size (random): 4051
Testing set size (random): 4051
Training set size (temporal): 12151
Validation set size (temporal): 4051
Testing set size (temporal): 4051
Total size: 20253
Class distribution for Temporal Training Split:
1    6283
0    5868
Name: Target_binary, dtype: int64
Class distribution for Temporal Validation Split:
0    2651
1    1400
Name: Target_binary, dtype: int64
Class distribution for Random Training Split:
0    6713
1    5438
Name: Target_binary, dtype: int64
Class distribution for Random Validation Split:
0    2190
1    1861

So the data after 2010 gets trained more evenly

"""



# FEATURE ENGINEERING FOR TRAINING DATA
# IDEAS 
# Geo features
# Average contract duration per Property
# Average contract duration per Complex
# Average contract duration per city

# Average contract duration per Region

# Property features
# Average contract duration per property type
# amount of bedrooms
# Rolling mean for size of property
# Rolling means for values of the property

# Lets start with these and add more if there is low correlation

def feature_engineering(df):
    features = pd.DataFrame(index=df.index)
    
    # Average Contract Duration per Property
    features['avg_contract_duration_per_property'] = df.groupby('VIBDRO_Huurobject_id')['Contract_duur'].transform('mean')
    
    # Average Contract Duration per Property Type
    features['avg_contract_duration_per_property_type'] = df.groupby('Omschrijving_Vastgoed')['Contract_duur'].transform('mean')
    
    # Average Contract Duration per Complex
    features['avg_contract_duration_per_complex'] = df.groupby('complexnummer')['Contract_duur'].transform('mean')
    
    # Average Contract Duration per City
    features['avg_contract_duration_per_city'] = df.groupby('Gemeente')['Contract_duur'].transform('mean')
    
    # Average Contract Duration per Region
    features['avg_contract_duration_per_region'] = df.groupby('regio')['Contract_duur'].transform('mean')

    # Rolling Mean for Property Value
    df = df.sort_values(['VIBDRO_Huurobject_id', 'Construction_year'])
    features['rolling_mean_property_value'] = df.groupby('VIBDRO_Huurobject_id')['WOZ waarde'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Amount of bedrooms in a house
    features['Aantal_slaapkamers'] = (
        (df['1e Slaapkamer'] > 0).astype(int) +
        (df['2e Slaapkamer'] > 0).astype(int) +
        (df['3e Slaapkamer'] > 0).astype(int)
    )
    
    def assign_huurklasse(markthuur):
        if pd.isna(markthuur):
            return 'Onbekend'
        elif markthuur <= 454.47:  # tot kwaliteitskortingsgrens
            return 'Goedkoop'
        elif markthuur <= 650.43:  # kwaliteitskortingsgrens - eerste aftoppingsgrens
            return 'klaliteitskorting Betaalbaar'
        elif markthuur <= 697.08:  # eerste aftoppingsgrens - tweede aftoppingsgrens
            return '1e aftoppingsgrens Betaalbaar'
        elif markthuur <= 879.66:  # tweede aftoppingsgrens - huurtoeslaggrens/DAEB-grens/liberalisatiegrens
            return 'Sociale hogere huur'
        else:
            return 'Vrije huur'

    features['Huurklasse'] = df['Markthuur'].apply(assign_huurklasse)

    df_with_features = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    return df_with_features

# Apply feature engineering to every set separately to avoid data leakage

train_data_temp_with_features = feature_engineering(train_data_temp)
train_data_rand_with_features = feature_engineering(train_data_rand)
validation_data_temp_with_features = feature_engineering(validation_data_temp)
validation_data_rand_with_features = feature_engineering(validation_data_rand)
test_data_temp_with_features = feature_engineering(test_data_temp)
test_data_rand_with_features = feature_engineering(test_data_rand)
unlabeled_data_with_features = feature_engineering(unlabeled_data)


# Function to plot correlation matrix, excluding categorical data that does not correlate numerically
def plot_correlation_matrix(df, title):
    plt.figure(figsize=(24, 20))
    
    # Selecting only numeric columns for correlation calculation in the plot
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    # Focus only on the correlation with target variable
    if 'Target_binary' in corr_matrix.columns:
        target_corr = corr_matrix[['Target_binary']].sort_values(by='Target_binary', ascending=False)
        
        # Plot the heatmap
        sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(title)
        plt.show()
    else:
        print("'Target_binary' not found in the correlation matrix.")

# Plot correlation matrix for temporal and random training sets
print("Correlation Matrix for Temporal Training Set") #I dont want to run this every time
plot_correlation_matrix(train_data_temp_with_features, "Correlation with Target - Temporal Training Set") #I dont want to run this every time

print("Correlation Matrix for Random Training Set")
plot_correlation_matrix(train_data_rand_with_features, "Correlation with Target - Random Training Set")

print("Overal correlation")
plot_correlation_matrix(labeled_data, "Correlation with Target - Random Training Set")

#House age high higest corr in the random split but not high corr in the temporal split
#Which means the temporal split managed to get out some of the temporal bias in the data
#Numerical variables arent very high in correlation, with highest corr features being around 20%, feature engineering resulted in weak features. 
#Rooms gave good correlation and WOZ-value, also the neighbourhood scored well. Maybe make more features around the house (e.g., amount of bedrooms)
#For the sake of consistency i want to drop the same columns in both training sets, so the experiment gauges the difference in splits
#Instead of the different in feature sets
#So im doing a comparison using a venn diagram
# Function to get columns with low correlation
def get_low_correlation_columns(df, threshold=0.1): #Threshold is 10% correlation 
    corr_matrix = df.corr()
    if 'Target_binary' in corr_matrix.columns:
        low_corr_cols = corr_matrix['Target_binary'][abs(corr_matrix['Target_binary']) < threshold].index.tolist()
        return low_corr_cols
    else:
        print("'Target_binary' not found in the correlation matrix.")
        return []

# Get low correlation columns for both random and temporal split training datasets
low_corr_temp = get_low_correlation_columns(train_data_temp_with_features)
low_corr_rand = get_low_correlation_columns(train_data_rand_with_features)

# Print the results
print(f"Low Correlation Columns (Temporal Split): {low_corr_temp}")
print(f"Low Correlation Columns (Random Split): {low_corr_rand}")

# Create a Venn diagram for visualization
plt.figure(figsize=(10, 6))
venn_labels = {'10': len(set(low_corr_temp)), '01': len(set(low_corr_rand)), '11': len(set(low_corr_temp) & set(low_corr_rand))}
venn2(subsets=venn_labels, set_labels=('Temporal Split', 'Random Split'))
plt.title("Venn Diagram of Low Correlation Columns (< 10% Correlation)")
plt.show()

#THere is some intersection
#Random split overal has higher correlation to features, which i think is due to the temporal bias in the data
#Which also means the temporal split did filter out some of the temporal bias, since temporal features are low % in the temporal split
#Columns below 10% correlation will be deleted to prevent overfitting

# Now doing the chi2 test for categorical features
# Define the categorical features in the dataset
categorical_features = [
    'Energielabel', 
    'Aardgasloze_woning',
    'Geen_deelname_energieproject',
    'Contractsoort', 
    'Reden_opzegging',
    'Omschrijving_Vastgoed',
    'Eengezins_Meergezins',
    'Gemeente',
    'regio',
    'Huurklasse'
]

# Store Chi-Squared test results separately
chi2_results_temp = {}
chi2_results_rand = {}

# Function to perform Chi-Squared test for a given DataFrame and target variable 
def perform_chi_squared_test(data, target_variable, results_dict):
    for feature in categorical_features:
        if feature in data.columns:  
            contingency_table = pd.crosstab(data[feature], data[target_variable])
            
            # Perform the Chi-Squared test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Store the results in the specified results dictionary
            results_dict[feature] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'expected_frequencies': expected
            }

# Perform Chi-Squared test on the temporal training dataset
print("Chi-Squared Test Results for Temporal Training Set")
perform_chi_squared_test(train_data_temp_with_features, 'Target_binary', chi2_results_temp)

# Convert results to a DataFrame for better readability
chi2_results_temp_df = pd.DataFrame(chi2_results_temp).T
print(chi2_results_temp_df)

# Perform Chi-Squared test on the random training dataset
print("Chi-Squared Test Results for Random Training Set")
perform_chi_squared_test(train_data_rand_with_features, 'Target_binary', chi2_results_rand)

# Convert results to a DataFrame for better readability
chi2_results_rand_df = pd.DataFrame(chi2_results_rand).T
print(chi2_results_rand_df)

# Function to interpret the Chi-Squared test results based on alpha == 0.05 to see if there is a significant relation or not.
def interpret_chi2_results(chi2_results_df, alpha=0.05):
    for feature, results in chi2_results_df.iterrows():
        chi2_stat = results['chi2_statistic']
        p_value = results['p_value']
        
        print(f"Feature: {feature}")
        print(f"Chi-Squared Statistic: {chi2_stat:.4f}")
        print(f"P-Value: {p_value:.4f}")
        
        if p_value < alpha:
            print("Conclusion: Reject the null hypothesis. There is a significant association between this feature and the target variable.")
        else:
            print("Conclusion: Fail to reject the null hypothesis. No significant association found.")
        print("-" * 50)

# Interpret Chi-Squared results for temporal training dataset
print("Interpreting Chi-Squared Test Results for Temporal Training Set")
interpret_chi2_results(chi2_results_temp_df)

# Interpret Chi-Squared results for random training dataset
print("Interpreting Chi-Squared Test Results for Random Training Set")
interpret_chi2_results(chi2_results_rand_df)


#Most categorical features are highly correlated with the target variable, That is good
#Going to drop low correlation columns 

columns_to_drop = ['Target','Ontvangstdatum_opzegging','Einddatum_contract','Contract_duur','Land', 'Energie_index', 
    'Marktwaarde', 'Totale punten (afgerond)',
    'Totale punten (onafgerond)','WOZ waarde per m2', 
    'WOZ waarde per m2 (WWS)', 'Woonkamer', 'Markthuur', 'Maximaal_redelijke_huur', 'Streefhuur', 
    'Year of demolition flag', 'EP2_waarde flag', 'Ontvangstdatum_opzegging flag', 
    'Reden_opzegging flag', 'Demolition_year', 'avg_contract_duration_per_property', 
    'avg_contract_duration_per_property_type', 'avg_contract_duration_per_complex', 
    'avg_contract_duration_per_city', 'avg_contract_duration_per_region', 'rolling_mean_property_value', 
    'Aantal_slaapkamers', 'Aardgasloze_woning','Geen_deelname_energieproject','Contractnummer','VIBDRO_Huurobject_id','Gemeente',
    'Woning_type','VERA_Type','Straat', 'Reden_opzegging','age_at_contract_start','age_bucket','age_bucket_imputed',
    'df_VIBPOBJREL_INTRENO','df_BUT000_BIRTHDT','Contract_duur flag', 'Contract_starting','Contract_starting_year','contract_year','birth_year',
    'Huurobject_y','Complex','VHE'] 


pd.set_option('display.max_columns', None)

# check to column names to see if i have missed anything
print(train_data_temp_with_features.columns.tolist())

# Apply column drop to all datasets seperately
train_data_temp_with_features = train_data_temp_with_features.drop(columns=columns_to_drop)
train_data_rand_with_features = train_data_rand_with_features.drop(columns=columns_to_drop)
validation_data_temp_with_features = validation_data_temp_with_features.drop(columns=columns_to_drop)
validation_data_rand_with_features = validation_data_rand_with_features.drop(columns=columns_to_drop)
test_data_temp_with_features = test_data_temp_with_features.drop(columns=columns_to_drop)
test_data_rand_with_features = test_data_rand_with_features.drop(columns=columns_to_drop)
unlabeled_data_with_features = unlabeled_data_with_features.drop(columns=columns_to_drop)

# Extract features (X) and target variable (y)
X_temp = train_data_temp_with_features.drop(columns=['Target_binary'])
y_temp = train_data_temp_with_features['Target_binary']

X_rand = train_data_rand_with_features.drop(columns=['Target_binary'])
y_rand = train_data_rand_with_features['Target_binary']

X_val_temp = validation_data_temp_with_features.drop(columns=['Target_binary'])
y_val_temp = validation_data_temp_with_features['Target_binary']

X_val_rand = validation_data_rand_with_features.drop(columns=['Target_binary'])
y_val_rand = validation_data_rand_with_features['Target_binary']

X_test_temp = test_data_temp_with_features.drop(columns=['Target_binary'])
y_test_temp = test_data_temp_with_features['Target_binary']

X_test_rand = test_data_rand_with_features.drop(columns=['Target_binary'])
y_test_rand = test_data_rand_with_features['Target_binary']

print(X_temp.shape)
print(X_rand.shape)
print(X_val_rand.shape)
print(X_val_temp.shape)
print(X_test_temp.shape)
print(X_test_rand.shape)


#Doing this for troubleshooting purposes, as it went wrong earlier 
for df in [X_temp, X_rand, X_val_temp, X_val_rand, X_test_temp, X_test_rand, unlabeled_data_with_features]:
    for col in df.select_dtypes(include=['datetime64']):
        df[col] = df[col].dt.year

# Going to do one hot encoding, i do it this way so all datasets have the same columns across, because earlier i did it wrong and when testing my model on the validation set it said columns where missing

# Identify categorical columns 
categorical_cols = X_temp.select_dtypes(include=['object']).columns

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#OneHotEncoder
encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)

#fitting it on temporal training data
encoder.fit(X_temp[categorical_cols])

#encoding 
X_temp_encoded = encoder.transform(X_temp[categorical_cols])
X_rand_encoded = encoder.transform(X_rand[categorical_cols])
X_val_temp_encoded = encoder.transform(X_val_temp[categorical_cols])
X_val_rand_encoded = encoder.transform(X_val_rand[categorical_cols])
X_test_temp_encoded = encoder.transform(X_test_temp[categorical_cols])
X_test_rand_encoded = encoder.transform(X_test_rand[categorical_cols])
unlabeled_data_encoded = encoder.transform(unlabeled_data_with_features[categorical_cols])

# turn back into dataframes
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

X_temp_encoded_df = pd.DataFrame(X_temp_encoded, columns=encoded_feature_names)
X_rand_encoded_df = pd.DataFrame(X_rand_encoded, columns=encoded_feature_names)
X_val_temp_encoded_df = pd.DataFrame(X_val_temp_encoded, columns=encoded_feature_names)
X_val_rand_encoded_df = pd.DataFrame(X_val_rand_encoded, columns=encoded_feature_names)
X_test_temp_encoded_df = pd.DataFrame(X_test_temp_encoded, columns=encoded_feature_names)
X_test_rand_encoded_df = pd.DataFrame(X_test_rand_encoded, columns=encoded_feature_names)
unlabeled_data_encoded_df = pd.DataFrame(unlabeled_data_encoded, columns=encoded_feature_names)


# Get non-categorical columns
non_categorical_cols = [col for col in X_temp.columns if col not in categorical_cols]

# Concatenate the non-categorical columns
X_temp_encoded_df = pd.concat([X_temp_encoded_df, X_temp[non_categorical_cols].reset_index(drop=True)], axis=1)
X_rand_encoded_df = pd.concat([X_rand_encoded_df, X_rand[non_categorical_cols].reset_index(drop=True)], axis=1)
X_val_temp_encoded_df = pd.concat([X_val_temp_encoded_df, X_val_temp[non_categorical_cols].reset_index(drop=True)], axis=1)
X_val_rand_encoded_df = pd.concat([X_val_rand_encoded_df, X_val_rand[non_categorical_cols].reset_index(drop=True)], axis=1)
X_test_temp_encoded_df = pd.concat([X_test_temp_encoded_df, X_test_temp[non_categorical_cols].reset_index(drop=True)], axis=1)
X_test_rand_encoded_df = pd.concat([X_test_rand_encoded_df, X_test_rand[non_categorical_cols].reset_index(drop=True)], axis=1)
unlabeled_data_encoded_df = pd.concat([unlabeled_data_encoded_df, unlabeled_data_with_features[non_categorical_cols].reset_index(drop=True)], axis=1)

print("Shapes after encoding:")
print("X_temp_encoded shape:", X_temp_encoded_df.shape)
print("X_rand_encoded shape:", X_rand_encoded_df.shape)
print("X_val_temp_encoded shape:", X_val_temp_encoded_df.shape)
print("X_val_rand_encoded shape:", X_val_rand_encoded_df.shape)
print("X_test_temp_encoded shape:", X_test_temp_encoded_df.shape)
print("X_test_rand_encoded shape:", X_test_rand_encoded_df.shape)
print("Unlabaled data shape:",unlabeled_data_encoded_df.shape)


# Save the training, validation, and test datasets for temporal split
X_temp_encoded_df.to_csv('X_train_temp.csv', index=False)
y_temp.to_csv('y_train_temp.csv', index=False)
X_val_temp_encoded_df.to_csv('X_val_temp.csv', index=False)
y_val_temp.to_csv('y_val_temp.csv', index=False)
X_test_temp_encoded_df.to_csv('X_test_temp.csv', index=False)
y_test_temp.to_csv('y_test_temp.csv', index=False)

# Save the training, validation, and test datasets for random split
X_rand_encoded_df.to_csv('X_train_rand.csv', index=False)
y_rand.to_csv('y_train_rand.csv', index=False)
X_val_rand_encoded_df.to_csv('X_val_rand.csv', index=False)
y_val_rand.to_csv('y_val_rand.csv', index=False)
X_test_rand_encoded_df.to_csv('X_test_rand.csv', index=False)
y_test_rand.to_csv('y_test_rand.csv', index=False)

#save the unlabeled dataset with preprocessing steps for the SSL experiment
unlabeled_data_encoded_df.to_csv('unlabeled_data_encoded_df.csv', index=False)
