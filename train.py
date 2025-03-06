import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

print("Starting training process...")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('heart_2022_no_nans.csv')

# Print initial info
print("\nInitial data info:")
print(f"Dataset shape: {df.shape}")
print("Target variable distribution:")
print(df['HadHeartAttack'].value_counts())

# Clean the data - remove any trailing commas or spaces
for col in df.select_dtypes(include=['object']).columns:
    print(f"Cleaning column: {col}")
    df[col] = df[col].astype(str).str.strip().str.rstrip(',')

# Simplify the target variable to binary
df['HadHeartAttack_Binary'] = df['HadHeartAttack'].apply(lambda x: 1 if x == 'Yes' else 0)

print("\nBinary target distribution:")
print(df['HadHeartAttack_Binary'].value_counts())

# Create a balanced dataset
yes_samples = df[df['HadHeartAttack_Binary'] == 1]
no_samples = df[df['HadHeartAttack_Binary'] == 0].sample(n=len(yes_samples) * 3, random_state=42)
balanced_df = pd.concat([yes_samples, no_samples])

print(f"\nBalanced dataset shape: {balanced_df.shape}")
print("Balanced target distribution:")
print(balanced_df['HadHeartAttack_Binary'].value_counts())

# Create a simple feature set with binary encoding for categorical variables
print("\nPreparing features...")
features_df = pd.DataFrame()

# Numeric features
features_df['BMI'] = balanced_df['BMI']
features_df['PhysicalHealthDays'] = balanced_df['PhysicalHealthDays']
features_df['MentalHealthDays'] = balanced_df['MentalHealthDays']
features_df['SleepHours'] = balanced_df['SleepHours']

# Binary encode Sex (1 for Male, 0 for Female)
features_df['Sex_Male'] = (balanced_df['Sex'] == 'Male').astype(int)

# Binary encode General Health (create multiple columns)
features_df['Health_Excellent'] = (balanced_df['GeneralHealth'] == 'Excellent').astype(int)
features_df['Health_VeryGood'] = (balanced_df['GeneralHealth'] == 'Very good').astype(int)
features_df['Health_Good'] = (balanced_df['GeneralHealth'] == 'Good').astype(int)
features_df['Health_Fair'] = (balanced_df['GeneralHealth'] == 'Fair').astype(int)
features_df['Health_Poor'] = (balanced_df['GeneralHealth'] == 'Poor').astype(int)

# Binary encode Yes/No features
features_df['PhysicalActivities'] = (balanced_df['PhysicalActivities'] == 'Yes').astype(int)
features_df['AlcoholDrinkers'] = (balanced_df['AlcoholDrinkers'] == 'Yes').astype(int)
features_df['HadStroke'] = (balanced_df['HadStroke'] == 'Yes').astype(int)
features_df['DifficultyWalking'] = (balanced_df['DifficultyWalking'] == 'Yes').astype(int)

# For diabetes, simplify to binary (any form of diabetes = 1)
features_df['HasDiabetes'] = balanced_df['HadDiabetes'].apply(
    lambda x: 1 if x.startswith('Yes') else 0
)

features_df['HadAsthma'] = (balanced_df['HadAsthma'] == 'Yes').astype(int)
features_df['HadKidneyDisease'] = (balanced_df['HadKidneyDisease'] == 'Yes').astype(int)
features_df['HadSkinCancer'] = (balanced_df['HadSkinCancer'] == 'Yes').astype(int)

# Target variable
y = balanced_df['HadHeartAttack_Binary'].values

print(f"Final feature matrix shape: {features_df.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Target distribution: {np.bincount(y)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    features_df.values, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set distribution: {np.bincount(y_train)}")
print(f"Testing set distribution: {np.bincount(y_test)}")

# Add synthetic high-risk positive examples
print("\nAdding synthetic high-risk positive examples...")
num_synthetic = 100
synthetic_high_risk = []

for _ in range(num_synthetic):
    # Create a high-risk profile that should predict heart attack
    sample = np.zeros(features_df.shape[1])
    
    # High BMI (30-40)
    sample[0] = np.random.uniform(30, 40)
    
    # Many physical health days (15-30)
    sample[1] = np.random.uniform(15, 30)
    
    # Many mental health days (10-30)
    sample[2] = np.random.uniform(10, 30)
    
    # Low sleep hours (3-5)
    sample[3] = np.random.uniform(3, 5)
    
    # Male (higher risk)
    sample[4] = 1
    
    # Poor health
    sample[5:10] = [0, 0, 0, 0, 1]  # Health_Poor = 1
    
    # No physical activity
    sample[10] = 0
    
    # Other risk factors
    sample[11:] = np.random.choice([0, 1], size=len(sample[11:]), p=[0.2, 0.8])
    
    synthetic_high_risk.append(sample)

# Add synthetic samples to training data
X_train = np.vstack([X_train, np.array(synthetic_high_risk)])
y_train = np.append(y_train, np.ones(num_synthetic))

print(f"New training set shape after synthetic data: {X_train.shape}")
print(f"New training set distribution: {np.bincount(y_train.astype(int))}")

# Train model with high weight on the positive class
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight={0: 1, 1: 10},  # Give even more weight to heart attack cases
    random_state=42,
    max_depth=10  # Limit depth to avoid overfitting
)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)

# Print results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Plot feature importance
feature_importance = pd.DataFrame({
    'feature': features_df.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()

# Test with a high-risk sample
print("\nTesting with a high-risk sample...")
high_risk_sample = np.array([
    38.5,  # BMI
    20.0,  # PhysicalHealthDays
    15.0,  # MentalHealthDays
    4.0,   # SleepHours
    1,     # Sex_Male
    0,     # Health_Excellent
    0,     # Health_VeryGood
    0,     # Health_Good
    0,     # Health_Fair
    1,     # Health_Poor
    0,     # PhysicalActivities
    1,     # AlcoholDrinkers
    1,     # HadStroke
    1,     # DifficultyWalking
    1,     # HasDiabetes
    1,     # HadAsthma
    1,     # HadKidneyDisease
    1      # HadSkinCancer
]).reshape(1, -1)

# Predict
high_risk_pred = rf_model.predict(high_risk_sample)[0]
high_risk_prob = rf_model.predict_proba(high_risk_sample)[0, 1]

print(f"High-risk prediction: {high_risk_pred} (1=Yes, 0=No)")
print(f"High-risk probability: {high_risk_prob:.4f}")

# Test with a low-risk sample
print("\nTesting with a low-risk sample...")
low_risk_sample = np.array([
    22.0,  # BMI
    0.0,   # PhysicalHealthDays
    0.0,   # MentalHealthDays
    8.0,   # SleepHours
    0,     # Sex_Male
    1,     # Health_Excellent
    0,     # Health_VeryGood
    0,     # Health_Good
    0,     # Health_Fair
    0,     # Health_Poor
    1,     # PhysicalActivities
    0,     # AlcoholDrinkers
    0,     # HadStroke
    0,     # DifficultyWalking
    0,     # HasDiabetes
    0,     # HadAsthma
    0,     # HadKidneyDisease
    0      # HadSkinCancer
]).reshape(1, -1)

# Predict
low_risk_pred = rf_model.predict(low_risk_sample)[0]
low_risk_prob = rf_model.predict_proba(low_risk_sample)[0, 1]

print(f"Low-risk prediction: {low_risk_pred} (1=Yes, 0=No)")
print(f"Low-risk probability: {low_risk_prob:.4f}")

# Save model and feature names
print("\nSaving model and feature information...")
model_data = {
    'model': rf_model,
    'features': list(features_df.columns)
}

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("Training complete! Model and visualizations have been saved.")