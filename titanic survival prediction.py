
print("THIS IS THE NEW FILE")
# TITANIC SURVIVAL PREDICTION USING MACHINE LEARNING
# CodSoft Machine Learning Internship
# Developed by: Ayush Mishra



# IMPORT LIBRARIES


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)



# LOAD DATASET


print("=" * 60)
print("Loading Titanic Dataset...")
print("=" * 60)

df = pd.read_csv("Dataset/Titanic-Dataset.csv")

print("\nDataset Loaded Successfully!")

print("\nFirst 5 Rows\n")
print(df.head())



# DATASET INFORMATION


print("\n" + "=" * 60)
print("Dataset Information")
print("=" * 60)

df.info()

print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)

print(df.isnull().sum())


# DATA CLEANING


print("\n" + "=" * 60)
print("Cleaning Dataset...")
print("=" * 60)


print("Before Age")

df["Age"] = df["Age"].fillna(df["Age"].median())

print("After Age")


df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


df = df.drop("Cabin", axis=1)

print("\nDataset Cleaned Successfully!")

print("\nMissing Values After Cleaning\n")

print(df.isnull().sum())
print("Reached EDA")

# Exploratory Data Analysis

# Survival Distribution

plt.figure(figsize=(6,4))

df["Survived"].value_counts().plot(kind="bar")

plt.title("Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

plt.savefig("screenshots/survival_distribution.png")

# plt.show(block=True)


# Survival by Gender

plt.figure(figsize=(6,4))

df.groupby("Sex")["Survived"].mean().plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.savefig("screenshots/survival_by_gender.png")

# plt.show()


# Age Distribution

plt.figure(figsize=(8,5))

plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.savefig("screenshots/age_distribution.png")

# plt.show()

# Feature Engineering

df = df.drop(["PassengerId", "Name", "Ticket"], axis=1)

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])

df["Embarked"] = encoder.fit_transform(df["Embarked"])

print("\nFeature Engineering Completed!")

# Features and Target

X = df.drop("Survived", axis=1)

y = df["Survived"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Model Training

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# Predictions

predictions = model.predict(X_test)

print("\nPredictions Generated Successfully!")

# Model Evaluation

accuracy = accuracy_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, predictions))   

# Confusion Matrix

plt.figure(figsize=(6,4))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.colorbar()

plt.savefig("screenshots/confusion_matrix.png")

# plt.show()

plt.close()

# Feature Importance

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8,5))

feature_importance.plot(kind="bar")

plt.title("Feature Importance")

plt.savefig("screenshots/feature_importance.png")

# plt.show()

plt.close()

print("\nProject Completed Successfully!")

print("Final Model Accuracy:", accuracy)
