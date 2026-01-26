
# Import
# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE  # For balancing classes

# Load Data
df = pd.read_csv("train.csv")
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nTarget Distribution:\n", df['Loan_Status'].value_counts())

# Exploratory Data Analysis (EDA)
cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

for col in cat_features:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='Loan_Status', data=df)
    plt.title(f"{col} vs Loan_Status")
    plt.legend(title='Loan_Status', labels=['Not Approved', 'Approved'])
    plt.show()

for col in num_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Loan_Status', y=col, data=df)
    plt.title(f"{col} vs Loan_Status")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"{col} Distribution")
    plt.show()

# Engineered features
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmount_log'] = np.log(df['LoanAmount'].replace(0, np.nan))
df['LoanAmount_log'] = df['LoanAmount_log'].fillna(df['LoanAmount_log'].median())

plt.figure(figsize=(6,4))
sns.boxplot(x='Loan_Status', y='TotalIncome', data=df)
plt.title("TotalIncome vs Loan_Status")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Loan_Status', y='LoanAmount_log', data=df)
plt.title("LoanAmount_log vs Loan_Status")
plt.show()

# Correlation matrix (numeric only)
plt.figure(figsize=(10,6))
numeric_cols = df.select_dtypes(include=np.number)
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Numeric Features Only)")
plt.show()

# Preprocessing
df = df.drop('Loan_ID', axis=1)

# Fill missing categorical values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Fill missing numerical values
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

# Encode categorical variables
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
df['Married'] = df['Married'].map({'Yes':1, 'No':0})
df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})
df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)
df['Dependents'] = df['Dependents'].astype(int)

# Train-Test Split
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'TotalIncome', 'LoanAmount_log']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE, class distribution:")
print(pd.Series(y_train_res).value_counts())

# Model Building - Random Forest
model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(X_train_res, y_train_res)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report (Random Forest with SMOTE):\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

#  Feature Importance Plot
importances = model.feature_importances_
feat_names = X.columns
feat_importance_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df)
plt.title("Feature Importance - Random Forest with SMOTE")
plt.show()
