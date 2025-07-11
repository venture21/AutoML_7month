

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 데이터 불러오기
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: train.csv 파일을 찾을 수 없습니다. 스크립트와 같은 디렉토리에 있는지 확인하세요.")
    exit()

# --- 데이터 전처리 ---
print("데이터 전처리를 시작합니다...")

# Age 결측치를 중앙값으로 대체
df['Age'].fillna(df['Age'].median(), inplace=True)

# Cabin 결측치 처리 -> Has_Cabin 특성 생성
df['Has_Cabin'] = df['Cabin'].notna().astype(int)

# Embarked 결측치를 최빈값으로 대체
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Sex 레이블 인코딩
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Embarked 원-핫 인코딩
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# FamilySize, IsAlone 특성 생성
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Title 특성 추출 및 인코딩
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df['Title'] = df['Title'].map(title_mapping)
df['Title'].fillna(0, inplace=True)

# 불필요한 특성 제거
df_processed = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
print("데이터 전처리 완료.")

# --- 데이터 시각화 ---
print("\n데이터 시각화를 시작합니다...")

# 1. 생존자 분포
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=df_processed)
plt.title('Survival Distribution (0 = No, 1 = Yes)')
plt.savefig('survival_distribution.png')
print("생존자 분포 차트를 'survival_distribution.png'로 저장했습니다.")

# 2. 상관관계 히트맵
plt.figure(figsize=(12, 10))
sns.heatmap(df_processed.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
print("상관관계 히트맵을 'correlation_heatmap.png'로 저장했습니다.")

# 3. Pclass별 생존율
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=df_processed)
plt.title('Survival Rate by Pclass')
plt.savefig('survival_by_pclass.png')
print("Pclass별 생존율 차트를 'survival_by_pclass.png'로 저장했습니다.")

# 4. 성별 생존율
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=df_processed)
plt.title('Survival Rate by Sex (0 = Male, 1 = Female)')
plt.xticks([0, 1], ['Male', 'Female'])
plt.savefig('survival_by_sex.png')
print("성별 생존율 차트를 'survival_by_sex.png'로 저장했습니다.")

# 5. 나이 분포별 생존율
plt.figure(figsize=(10, 6))
sns.histplot(data=df_processed, x='Age', hue='Survived', multiple='stack', kde=True)
plt.title('Age Distribution by Survival')
plt.savefig('age_distribution.png')
print("나이 분포별 생존율 차트를 'age_distribution.png'로 저장했습니다.")
print("데이터 시각화 완료.")


# --- 모델링 ---
print("\n모델 학습 및 평가를 시작합니다...")

# 특성(X)과 타겟(y) 분리
X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 초기화
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# 모델 학습 및 평가
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

print("모델 학습 및 평가 완료.")

# --- 하이퍼파라미터 튜닝 (Gradient Boosting) ---
print("\nGradient Boosting 모델 하이퍼파라미터 튜닝을 시작합니다...")

# 튜닝할 파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# GridSearchCV 객체 생성
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=3, # 3-fold cross-validation
    scoring='accuracy',
    n_jobs=-1 # 모든 CPU 코어 사용
)

# 그리드 서치 수행
grid_search.fit(X_train, y_train)

# 최적 파라미터 및 점수 출력
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 교차 검증 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 테스트 데이터 예측 및 평가
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print(f"튜닝 후 Gradient Boosting 최종 정확도: {tuned_accuracy:.4f}")
print("하이퍼파라미터 튜닝 완료.")

