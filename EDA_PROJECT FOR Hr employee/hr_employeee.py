import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("hr_employee.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns in Dataset:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())

print("\nUnique values in each column:")
print(df.nunique())

df = df.drop(columns=[
    'EmployeeCount',
    'EmployeeNumber',
    'Over18',
    'StandardHours'
])

print("\nColumns after dropping unnecessary columns:")
print(df.columns)

plt.figure()
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Distribution")
plt.show()

plt.figure()
sns.histplot(df['Age'], bins=20)
plt.title("Age Distribution of Employees")
plt.show()

plt.figure()
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.show()

plt.figure()
sns.countplot(x='Department', data=df)
plt.title("Department Distribution")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(x='JobRole', data=df)
plt.title("Job Role Distribution")
plt.xticks(rotation=90)
plt.show()


plt.figure()
sns.countplot(x='Attrition', hue='Gender', data=df)
plt.title("Attrition by Gender")
plt.show()

plt.figure()
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.show()

plt.figure()
sns.histplot(df['MonthlyIncome'], bins=20)
plt.title("Monthly Income Distribution")
plt.show()

corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=False)
plt.title("Correlation Heatmap")
plt.show()

plt.figure()
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title("Overtime vs Attrition")
plt.show()

print("\nFinal Dataset Shape After Cleaning:")
print(df.shape)