#!/usr/bin/env python
# coding: utf-8

# # Midterm Project

# Dataset from kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# ## Data cleaning and preparation

# - Downloading the dataset
# - Doing the train/validation/test split

df = pd.read_csv(r"C:\Users\pc\Downloads\archive (10)\heart.csv")

df

df.describe().round()

df.info()
# Display the first 5 rows
df.head()

df.columns

df.isna().sum()

duplicates = df[df.duplicated()]

print(duplicates)


# There are 723 duplicate rows in this dataset, and we will drop them so our prediction will not be biased or overfitted.

df = df.drop_duplicates()

df.shape

print(df.columns)
print()
print(df.dtypes)


# ## EDA

sns.histplot(df['age'] )
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(df['chol'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Cholesterol Levels')
plt.xlabel('Cholesterol (mg/dL)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

avg_chol_by_age = df.groupby('age')['chol'].mean()
plt.figure(figsize=(10, 6))
plt.plot(avg_chol_by_age.index, avg_chol_by_age.values, marker='o', linestyle='-', color='coral')
plt.title('Average Cholesterol by Age')
plt.xlabel('Age (years)')
plt.ylabel('Average Cholesterol (mg/dL)')
plt.grid(True)
plt.show()


# **Target Distribution**

target = df["target"].value_counts()
print(target)

sick = (target[0]/df["target"].count()*100).round(2)
hearty = (target[1]/df["target"].count()*100).round(2)

print("Percentage of patience without heart problems:",hearty)
print()
print("Percentage of patience with heart problems:", sick)


df.head()

plt.figure(figsize = (10,6))

plt.subplot(1,2,1)
sns.countplot(x = "target", data = df)
plt.title("Heart Disease Target")
plt.xlabel("Target")
plt.ylabel("Num of patient")


df["sex"].unique()


# 1 = Male Patient
# 
# 0 = Female Patient

sex = df["sex"].value_counts()
print(sex)

# Count the number of male and female patients
sex_counts = df["sex"].value_counts()

# Calculate percentages
male = (sex_counts[1] / df["sex"].count() * 100).round(2)
female = (sex_counts[0] / df["sex"].count() * 100).round(2)

# Print results
print("Percentage of female patients:", female)
print()
print("Percentage of male patients:", male)

# Count the number of male and female patients
sex_counts = df["sex"].value_counts()

# Calculate percentages
male = (sex_counts[1] / df["sex"].count() * 100).round(2)
female = (sex_counts[0] / df["sex"].count() * 100).round(2)

# Print results
print("Percentage of female patients:", female)
print()
print("Percentage of male patients:", male)

pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,8))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# **Correlation: Heatmap**

plt.figure(figsize = (10,8))
sns.heatmap(df.corr(), annot = True,cmap='coolwarm')
plt.show()


corr = df.corr(numeric_only=True)["target"].sort_values()
print(corr)


# ## Model Training & Evaluation

# **Split dataset into features (x) and target (y)**

from sklearn.model_selection import train_test_split

# Do train/validation/test split with 60%/20%/20% distribution.
# Use the train_test_split function and set the random_state parameter to 1.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[28]:


df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

df_train.head()


# Let's define y:

y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values


# Let's drop the target variable from our dataframes:

del df_train["target"]
del df_val["target"]
del df_test["target"]


# Now we are ready to train a model.

# ## Logistic Regression Model

#Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(df_train_scaled,y_train)
y_pred_lr = lr.predict(df_test_scaled)
y_proba_lr = lr.predict_proba(df_test_scaled)[:,1]

#Model Performance Metrics

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

precision_lr = precision_score(y_test,y_pred_lr)
recall_lr = recall_score(y_test,y_pred_lr)
f1_lr = f1_score(y_test,y_pred_lr)
roc_auc_lr = roc_auc_score(y_test,y_proba_lr)

print("Precision_score :",round(precision_lr,2))
print("Recall_score :", round(recall_lr,2))
print("F1_score :",round(f1_lr,2))
print("Roc_auc_score :", round(roc_auc_lr,2))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

com_lr = confusion_matrix(y_test,y_pred_lr)
print("Confusion Matrix:\n", com_lr)

disp = ConfusionMatrixDisplay(confusion_matrix=com_lr, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


# ## Random Forest Model

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(bootstrap = True, max_depth= 5, min_samples_leaf= 4, min_samples_split= 2, n_estimators= 300)
rf.fit(df_train_scaled, y_train)
y_pred_rf = rf.predict(df_test_scaled)
y_probs_rf = rf.predict_proba(df_test_scaled)[:, 1]

# model evaluation

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

precision_rfc = precision_score(y_test,y_pred_rf)
recall_rfc = recall_score(y_test,y_pred_rf)
f1_rfc = f1_score(y_test,y_pred_rf)
roc_auc_rfc = roc_auc_score(y_test,y_probs_rf)

print("Precision Score :",round(precision_rfc,2))
print("Recall Score :",round(recall_rfc,2))
print("F1 Score :",round(f1_rfc,2))
print("Roc Auc Score :",round(roc_auc_rfc,2))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))


# ## XGBoost 

from xgboost import XGBClassifier
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 42, colsample_bytree= 0.7, gamma = 0.2, learning_rate = 0.01, max_depth= 7, n_estimators= 200, subsample = 0.7)
xgb.fit(df_train_scaled, y_train)
y_pred_xgb = xgb.predict(df_test_scaled)
y_proba_xgb = xgb.predict_proba(df_test_scaled)[:, 1]

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

precision_xgb = precision_score(y_test,y_pred_xgb)
recall_xgb = recall_score(y_test,y_pred_xgb)
f1_xgb = f1_score(y_test,y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test,y_proba_xgb)

print("Precision Score :",round(precision_xgb,2))
print("Recall Score :",round(recall_xgb,2))
print("F1 Score :",round(f1_xgb,2))
print("Roc Auc Score :",round(roc_auc_xgb,2))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))

# ## SVM

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline: scale features then apply SVM
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, C=1.0, gamma='scale'))

# Fit the model
svm_model.fit(df_train_scaled, y_train)

# Predict class labels
y_pred_svm = svm_model.predict(df_test_scaled)

# Predict probabilities
y_probs_svm = svm_model.predict_proba(df_test_scaled)[:, 1]


# SVM performance metrics
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_probs_svm)

print("Precision Score :",round(precision_svm,2))
print("Recall Score :",round(recall_svm,2))
print("F1 Score :",round(f1_svm,2))
print("Roc Auc Score :",round(roc_auc_svm,2))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))


# ## Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

# Initialize the model with optional hyperparameters
dtc = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=4)

# Fit the model
dtc.fit(df_train_scaled, y_train)

# Predict class labels
y_pred_dtc = dtc.predict(df_test_scaled)

# Predict probabilities (for ROC AUC)
y_probs_dtc = dtc.predict_proba(df_test_scaled)[:, 1]

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

precision_dtc = precision_score(y_test, y_pred_dtc)
recall_dtc = recall_score(y_test, y_pred_dtc)
f1_dtc = f1_score(y_test, y_pred_dtc)
roc_auc_dtc = roc_auc_score(y_test, y_probs_dtc)

print("Precision Score :",round(precision_dtc,2))
print("Recall Score :",round(recall_dtc,2))
print("F1 Score :",round(f1_dtc,2))
print("Roc Auc Score :",round(roc_auc_dtc,2))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dtc))

models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM', 'Decision Tree']
precision = [0.82, 0.80, 0.78, 0.78, 0.83]
recall = [0.77, 0.80, 0.83, 0.83, 0.69]
f1 = [0.79, 0.80, 0.81, 0.81, 0.75]
roc_auc = [0.85, 0.89, 0.87, 0.84, 0.80]

metrics_df = pd.DataFrame({
    'Model': models,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC-AUC': roc_auc
})

metrics_df.index = range(1, len(metrics_df) + 1)

metrics_df


# Save the model

import pickle

# Save the model to a file
with open('heart_model.pkl', 'wb') as file:
    pickle.dump(lr, file)
