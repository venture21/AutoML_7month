
import pandas as pd
import numpy as np

# 데이터 불러오기
train_df = pd.read_csv('C:/Users/park0/github/AutoML_7month/train.csv')
test_df = pd.read_csv('C:/Users/park0/github/AutoML_7month/test.csv')

# 테스트 데이터셋의 PassengerId를 나중에 사용하기 위해 저장
test_passenger_id = test_df['PassengerId']

# 데이터 합치기 (전처리를 일관성 있게 적용하기 위함)
combined_df = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)

# 결측치 처리
# Age: 중앙값으로 채우기
combined_df['Age'].fillna(combined_df['Age'].median(), inplace=True)

# Embarked: 최빈값으로 채우기
combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)

# Fare: 중앙값으로 채우기
combined_df['Fare'].fillna(combined_df['Fare'].median(), inplace=True)

# 피처 엔지니어링
# Sex: 숫자형으로 변환
combined_df['Sex'] = combined_df['Sex'].map({'male': 0, 'female': 1})

# Embarked: 원-핫 인코딩
combined_df = pd.get_dummies(combined_df, columns=['Embarked'], prefix='Embarked')

# Name에서 Title 추출
combined_df['Title'] = combined_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
rare_titles = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 
    'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer', 'Dr': 'Officer', 'Rev': 'Officer',
    'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Dona': 'Royalty', 'Sir': 'Royalty', 'the Countess': 'Royalty', 'Lady': 'Royalty'
}
combined_df['Title'] = combined_df['Title'].replace(rare_titles)
combined_df = pd.get_dummies(combined_df, columns=['Title'], prefix='Title')

# FamilySize 생성
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1

# 불필요한 피처 제거
combined_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)

# 데이터 분리
train_preproc = combined_df.iloc[:len(train_df)]
test_preproc = combined_df.iloc[len(train_df):]

# train_preproc에 Survived 컬럼 다시 추가
train_preproc['Survived'] = train_df['Survived']

# 전처리된 데이터 저장
train_preproc.to_csv('C:/Users/park0/github/AutoML_7month/train_preproc.csv', index=False)
test_preproc.to_csv('C:/Users/park0/github/AutoML_7month/test_preproc.csv', index=False)

print("train_preproc.csv와 test_preproc.csv 파일이 생성되었습니다.")
