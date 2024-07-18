import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Đọc dữ liệu từ tệp CSV
loan_df = pd.read_csv(r"C:\Users\ACER\Downloads\project\final\dataset.csv")

# Hiển thị 5 dòng đầu của DataFrame
print(loan_df.head())

# Hiển thị thông tin về DataFrame
loan_df.info()

# Phân tích thống kê mô tả
loan_df.describe()

# Xử lý missing values
loan_df['ApplicantIncome'] = pd.to_numeric(loan_df['ApplicantIncome'], errors='coerce')
loan_df['CoapplicantIncome'] = pd.to_numeric(loan_df['CoapplicantIncome'], errors='coerce')
loan_df['Total_Income'] = loan_df['ApplicantIncome'] + loan_df['CoapplicantIncome']

# Điền giá trị thiếu cho các biến số bằng giá trị trung bình
loan_df['LoanAmount'] = loan_df['LoanAmount'].fillna(loan_df['LoanAmount'].mean())
loan_df['Loan_Amount_Term'] = loan_df['Loan_Amount_Term'].fillna(loan_df['Loan_Amount_Term'].mean())
loan_df['Credit_History'] = loan_df['Credit_History'].fillna(loan_df['Credit_History'].mean())

# Điền giá trị thiếu cho các biến phân loại bằng giá trị mode
for column in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
    loan_df[column].fillna(loan_df[column].mode()[0], inplace=True)

# Chuyển đổi biến phân loại thành dạng số
loan_df['Loan_Status'] = np.where(loan_df['Loan_Status'] == 'Y', 1, 0)

# Giải quyết vấn đề mất cân bằng dữ liệu
df_majority = loan_df[loan_df['Loan_Status'] == 1]
df_minority = loan_df[loan_df['Loan_Status'] == 0]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
loan_df = pd.concat([df_majority, df_minority_upsampled])

# Chuyển đổi các biến phân loại thành biến giả
loan_df = pd.get_dummies(loan_df, drop_first=True)

# Chia tập train và test
X = loan_df.drop(columns=['Loan_Status'])
y = loan_df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Xử lý categorical features và scale numerical features trong pipeline
categorical_features = [col for col in X.columns if X[col].dtype == 'uint8']
numeric_features = ['Credit_History', 'Loan_Amount_Term', 'Total_Income']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)
    ]
)

# Random Forest Classifier
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', random_forest_model)
])

random_forest_pipeline.fit(X_train, y_train)

# Lưu mô hình đã huấn luyện
joblib.dump(random_forest_pipeline, 'random_forest_pipeline.pkl')

# Đánh giá mô hình
y_pred = random_forest_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
