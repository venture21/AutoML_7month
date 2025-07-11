import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# 데이터 불러오기
titanic_df = pd.read_csv('train.csv')

# 데이터 전처리
def preprocess_data(df):
    # Age 결측치를 중앙값으로 채우기
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Embarked 결측치를 최빈값으로 채우기
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Cabin 열 삭제
    df.drop('Cabin', axis=1, inplace=True)

    # Sex 열을 숫자형으로 변환
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Embarked 열을 원-핫 인코딩
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)

    # 불필요한 열 삭제
    df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    return df

titanic_processed = preprocess_data(titanic_df.copy())

# 전처리된 데이터를 CSV 파일로 저장
titanic_processed.to_csv('preprocessed_titanic.csv', index=False)

print("preprocessed_titanic.csv 파일이 저장되었습니다.")


# 시각화
def visualize_data(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Survived', data=df)
    plt.title('Survival Distribution')
    plt.savefig('survival_distribution.png')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title('Survival by Sex')
    plt.savefig('survival_by_sex.png')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title('Survival by Pclass')
    plt.savefig('survival_by_pclass.png')

    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.savefig('age_distribution.png')

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')

    print("시각화 파일들이 저장되었습니다.")

visualize_data(titanic_processed)


# 모델링
def model_data(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 모델 생성
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 교차 검증
    scores = cross_val_score(model, X, y, cv=5)
    print(f'Cross-validation scores: {scores}')
    print(f'Average cross-validation score: {scores.mean():.2f}')

    # 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 평가
    print('\nModel Performance:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

model_data(titanic_processed)