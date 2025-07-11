import pandas as pd

# Corrected file paths with escaped backslashes
xls_path = "C:\\Users\\park0\\github\\AutoML_7month\\titanic.xls"
test_path = "C:\\Users\\park0\\github\\AutoML_7month\\test.csv"
output_path = "C:\\Users\\park0\\github\\AutoML_7month\\test_with_survived.csv"

# titanic.xls 파일을 읽어옵니다.
xls_df = pd.read_excel(xls_path)

# test.csv 파일을 읽어옵니다.
test_df = pd.read_csv(test_path)

# titanic.xls의 이름 목록을 가져옵니다.
xls_names = xls_df['name'].unique()

# test.csv의 이름과 비교하여 'Survived' 컬럼을 추가합니다.
test_df['Survived'] = test_df['Name'].isin(xls_names).astype(int)

# 결과를 새로운 CSV 파일에 저장합니다.
test_df.to_csv(output_path, index=False)

print(f"'{output_path}' 파일이 생성되었습니다.")