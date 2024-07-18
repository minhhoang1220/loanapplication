#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Đọc dữ liệu từ tệp CSV
loan_df = pd.read_csv(r"C:\Users\ACER\Downloads\project\final\dataset.csv")

# Hiển thị 5 dòng đầu của DataFrame
print(loan_df.head())

# Hiển thị thông tin về DataFrame
loan_df.info()

# Phân tích thống kê mô tả
loan_df.describe()

# Vẽ biểu đồ countplot
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.subplot(231)
sns.countplot(x="Gender", hue='Loan_Status', data=loan_df)
plt.subplot(232)
sns.countplot(x="Married", hue='Loan_Status', data=loan_df)
plt.subplot(233)
sns.countplot(x="Education", hue='Loan_Status', data=loan_df)
plt.subplot(234)
sns.countplot(x="Self_Employed", hue='Loan_Status', data=loan_df)
plt.subplot(235)
sns.countplot(x="Dependents", hue='Loan_Status', data=loan_df)
plt.subplot(236)
sns.countplot(x="Property_Area", hue='Loan_Status', data=loan_df)
plt.show()

bins = np.linspace(loan_df.ApplicantIncome.min(), loan_df.ApplicantIncome.max(), 12)
graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
graph.map(plt.hist, "ApplicantIncome", bins=bins, ec="k")
graph.axes[-1].legend()

bins = np.linspace(loan_df.Loan_Amount_Term.min(), loan_df.Loan_Amount_Term.max(), 12)
graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
graph.map(plt.hist, "Loan_Amount_Term", bins=bins, ec="k")
graph.axes[-1].legend()

bins = np.linspace(loan_df.CoapplicantIncome.min(), loan_df.CoapplicantIncome.max(), 12)
graph = sns.FacetGrid(loan_df, col="Gender", hue="Loan_Status", palette="Set2", col_wrap=2)
graph.map(plt.hist, "CoapplicantIncome", bins=bins, ec="k")
graph.axes[-1].legend()

plt.show()

#Scatter plot
sns.set(rc={'figure.figsize':(11.7,12.27)})
plt.subplot(231)
sns.scatterplot(x="ApplicantIncome", y="LoanAmount", hue='Loan_Status', data=loan_df)
plt.title('by Loan Status')

plt.subplot(232)
sns.scatterplot(x="ApplicantIncome", y="LoanAmount", hue='Gender', data=loan_df)
plt.title('by Gender')

plt.subplot(233)
sns.scatterplot(x="ApplicantIncome", y="LoanAmount", hue='Married', data=loan_df)
plt.title('by Marital Status')

plt.subplot(234)
sns.scatterplot(x="ApplicantIncome", y="LoanAmount", hue='Education', data=loan_df)
plt.title('by Education')

plt.subplot(235)
sns.scatterplot(x="ApplicantIncome", y="LoanAmount", hue='Self_Employed', data=loan_df)
plt.title('by Self Employment')

plt.subplot(236)
sns.scatterplot(x="ApplicantIncome", y="LoanAmount", hue='Dependents', data=loan_df)
plt.title('by Dependents')

plt.show()

#Pie chart
loan_term_count = loan_df.groupby(['Gender', 'Loan_Status'])['Loan_Amount_Term'].count().unstack()
loan_term_count.plot(kind='pie', subplots=True, figsize=(10, 6), autopct='%1.1f%%')
plt.title('Loan Amount Term Distribution by Gender and Loan Status')
loan_term_count = loan_df.groupby(['Married', 'Loan_Status'])['Loan_Amount_Term'].count().unstack()
loan_term_count.plot(kind='pie', subplots=True, figsize=(10, 6), autopct='%1.1f%%')
plt.title('Loan Amount Term Distribution by Marital Status and Loan Status')

plt.show()

# Lựa chọn các cột không phải là số
categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History', 'Loan_Amount_Term']
categorical_df = loan_df[categorical_columns]

numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'Total_Income']

# Xử lý missing values
loan_df['ApplicantIncome'] = pd.to_numeric(loan_df['ApplicantIncome'], errors='coerce')
loan_df['CoapplicantIncome'] = pd.to_numeric(loan_df['CoapplicantIncome'], errors='coerce')
loan_df['Total_Income'] = loan_df['ApplicantIncome'] + loan_df['CoapplicantIncome']
# Chỉ chọn các cột số
numeric_columns = loan_df.select_dtypes(include=[np.number])

# Tính trung bình cho từng cột số và điền vào các giá trị thiếu
missing_values = loan_df.isnull().sum()
percentage_missing = (loan_df.isnull().sum() / len(loan_df)) * 100

# Drop rows with any missing values
df_cleaned = loan_df.dropna()

# Fill missing values with mean
df_filled = loan_df.fillna(loan_df.mean(numeric_only=True), inplace=True)

loan_transformed = loan_df.copy()

# Tính trung bình cho từng cột số và điền vào các giá trị thiếu
for column in numeric_columns.columns:
    loan_df[column] = loan_df[column].fillna(loan_df[column].mean())

for feature in categorical_columns:
  loan_transformed[feature] = np.where(loan_transformed[feature].isnull(),
                                       loan_transformed[feature].mode(),
                                       loan_transformed[feature])
# với những cột giá trị dạng số mà có ô bị thiếu thì sẽ fill bằng giá trị median(aka giá trị xuất hiện ở giữa(khác vs mean đấy))
for feature in numerical_columns:
  loan_transformed[feature] = np.where(loan_transformed[feature].isnull(),
                                       int(loan_transformed[feature].median()),
                                       loan_transformed[feature])


loan_transformed.isnull().sum()

for feature in numerical_columns:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.boxplot(loan_df[feature])
    plt.title(f'Before Log-Transformation: {feature}')
    plt.xlabel('Dataset')
    plt.ylabel(feature)
    plt.subplot(1, 2, 2)
    plt.boxplot(np.log1p(loan_df[feature]))
    plt.title(f'After Log-Transformation: {feature}')
    plt.xlabel('Dataset')
    plt.ylabel(f'Log({feature} + 1)')
    plt.tight_layout()
    plt.show()

for feature in numerical_columns:
    loan_transformed[feature] = np.log1p(loan_transformed[feature])

# Chuyển đổi biến phân loại thành dạng số
loan_transformed['Loan_Status'] = np.where(loan_transformed['Loan_Status'] == 'Y', 1, 0)
loan_transformed = pd.get_dummies(loan_transformed, drop_first=True)

minority_class = loan_df['Loan_Status'].value_counts().sort_values().index[0]

# Resample lớp thiểu số (Loan_Status == 'N') để có cùng số lượng mẫu với lớp đa số (Loan_Status == 'Y')
df_resampled = pd.concat([
    resample(loan_df[loan_df['Loan_Status'] == 'N'], replace=True, n_samples=len(loan_df[loan_df['Loan_Status'] == 'Y'])),
    loan_df[loan_df['Loan_Status'] == 'Y']
])
# Kiểm tra mức độ cân bằng dữ liệu
print(df_resampled['Loan_Status'].value_counts())
sns.countplot(x='Loan_Status', data=df_resampled)
plt.show()

# Chia tập train và test
X = df_resampled[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']]
y = df_resampled['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Xử lý categorical features và scale numerical features trong pipeline
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numeric_features = ['Credit_History', 'Loan_Amount_Term']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)

# Logistic Regression Model
logistic_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, penalty='l2', solver='lbfgs'))
])
logistic_scores = cross_val_score(logistic_model, X_train, y_train, cv=5)
accuracy_logistic = np.mean(logistic_scores)
print("Logistic Regression Model Accuracy:", accuracy_logistic)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_logistic))

# Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', decision_tree_model)
])
decision_tree_scores = cross_val_score(decision_tree_pipeline, X_train, y_train, cv=5)
accuracy_decision_tree = np.mean(decision_tree_scores)
print("Decision Tree Model Accuracy:", accuracy_decision_tree)
decision_tree_pipeline.fit(X_train, y_train)
y_pred_decision_tree = decision_tree_pipeline.predict(X_test)
print("Classification Report for Decision Tree:")
print(classification_report(y_test, y_pred_decision_tree))

# Gradient Boosting Classifier
gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gradient_boosting_model)
])
gradient_boosting_scores = cross_val_score(gradient_boosting_pipeline, X_train, y_train, cv=5)
accuracy_gradient_boosting = np.mean(gradient_boosting_scores)
print("Gradient Boosting Model Accuracy:", accuracy_gradient_boosting)
gradient_boosting_pipeline.fit(X_train, y_train)
y_pred_gradient_boosting = gradient_boosting_pipeline.predict(X_test)
print("Classification Report for Gradient Boosting:")
print(classification_report(y_test, y_pred_gradient_boosting))

# Random Forest Classifier
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', random_forest_model)
])
random_forest_scores = cross_val_score(random_forest_pipeline, X_train, y_train, cv=5)
accuracy_random_forest = np.mean(random_forest_scores)
print("Random Forest Model Accuracy:", accuracy_random_forest)
random_forest_pipeline.fit(X_train, y_train)
y_pred_random_forest = random_forest_pipeline.predict(X_test)
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_random_forest))

# Dự đoán trạng thái vay dựa trên dữ liệu người dùng
def predict_loan_status(model):
    print("Please enter the following information:")
    gender = input("Gender (Male/Female): ").capitalize()
    married = input("Married (Yes/No): ").capitalize()
    dependents = input("Dependents (1-Yes/0-No): ")
    education = input("Education (Graduate/Not Graduate): ").capitalize()
    self_employed = input("Self Employed (Yes/No): ").capitalize()
    property_area = input("Property Area (Urban/Rural/Semiurban): ").capitalize()
    credit_history = float(input("Credit History (1.0/0.0): "))
    loan_amount_term = float(input("Loan Amount Term: "))

    user_data = pd.DataFrame({'Gender': [gender],
                              'Married': [married],
                              'Dependents': [dependents],
                              'Education': [education],
                              'Self_Employed': [self_employed],
                              'Property_Area': [property_area],
                              'Credit_History': [credit_history],
                              'Loan_Amount_Term': [loan_amount_term]})

    predicted_result = model.predict(user_data)
    print("\nPredicted Loan Status:", predicted_result[0])

# Dự đoán trạng thái vay dựa trên dữ liệu người dùng sử dụng Random Forest model
predict_loan_status(random_forest_pipeline)
