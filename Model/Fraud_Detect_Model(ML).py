# This ipynb aims to find a good model to detect Fraud cases.
# 이 ipynb 파일은 부정 거래를 탐지하는 적절한 모형을 찾는 것에 목적을 두고 있습니다.


# Data load 데이터 확인
import pandas as pd

df = pd.read_csv('../data/Fraud_Detection_sql.csv')
df.info()


# ready for get_dummies() 카테고리 변수 변환을 위한 작업
df['TRANS_MONTH'] = df['TRANS_MONTH'].astype(str)
df['TRANS_DAY_SIMPLIFIED'] = df['TRANS_DAY_SIMPLIFIED'].astype(str)
df['TRANS_HOUR_SIMPLIFIED'] = df['TRANS_HOUR_SIMPLIFIED'].astype(str)
df['CATEGORY'] = df['CATEGORY'].astype(str)
df['STATE'] = df['STATE'].astype(str)

# remove unnecessary columns for regression 불필요한 컬럼 제거
df = df.drop(columns='TRANS_YEAR')  
df = df.drop(columns='TRANS_DAY')
df = df.drop(columns='TRANS_HOUR')
df = df.drop(columns='CC_NUM')
df = df.drop(columns='AMT')

# those three requires too many columns for model  이 3개의 변수는 모형에 지나치게 많은 컬럼을 생성한다.
df = df.drop(columns='CITY')
df = df.drop(columns='JOB')
df = df.drop(columns='STATE')

# Unlike the AI model, the ML model uses those two columns below 
# 인공지능 모형과 달리 머신러닝 모형은 아래 두 컬럼을 사용한다.
# df = df.drop(columns='CITY_POP')
# df = df.drop(columns='CATEGORY')


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Data encoding through get_dummies() 범주형 변수를 원-핫 인코딩
df_encoded = pd.get_dummies(df)


x = df_encoded.drop('IS_FRAUD', axis=1)
y = df['IS_FRAUD']


# Data split 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)


# SMOTE for train sets only 학습 데이터에만 적용
smote = SMOTE(sampling_strategy={1: 800000}, random_state=10)
x_train_over, y_train_over = smote.fit_resample(x_train, y_train)


# Since total numeber is 1.6M, fraud case around 9000 data needs to be oversampled to its half, 0.8M.  
# 전체 데이터가 160만개이므로 부정거래 케이스를 담은 약 9000개의 데이터를 이것의 절반인 약 80만으로 확장한다.



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define data set with auto-detect categorical features  데이터셋 생성 및 카테고리형 변수 자동 감지
train_data = lgb.Dataset(x_train_over, label=y_train_over, categorical_feature='auto')

# LightGBM 
# Those parameters were from several trials and errors. 이 모형의 파라미터는 수동으로 시행착오를 겪으며 설정되었다.
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 100,
    'min_child_samples': 40,  # min_data_in_leaf 
    'reg_alpha': 2,  # lambda_l1 
    'reg_lambda': 4,  # lambda_l2 
    'colsample_bytree': 0.8,  # feature_fraction 
    'learning_rate': 0.5,
    'verbose':0
    
}
# Note that train_data is only chosen!  훈련 데이터만 선택됨에 유의한다!
model = lgb.train(params, train_data, num_boost_round=300)

# Prediction for test data  테스트 데이터에 대한 예측
y_pred = model.predict(x_test)

# Transform the prediction into binary numbers  예측값을 이진 분류로 변환
y_pred_binary = [1 if pred >= 0.326 else 0 for pred in y_pred]

# "Confusion Matrix 혼동행렬
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix(혼동행렬):")
print(conf_matrix)

# Model Evaluation 모델 평가
print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_binary))


# Despite 99 accuracy in the test set, the real point is the f1-score considering both precision and recall. This model can __detect fraud with 87%__ probability.  
# test set에 나오는 점수는 99점으로 준수해보이지만 본질적인 것은 1을 만났을 때 제대로 감지하는 지 여부다. 이 모형은 __87%확률로 부정거래를 감지__ 한다.

# Feature importance shows AMT, hour(3), category(2), AGE, day(2), month(12) has comparatively strong tendencies.  
# 이 그래프 결과는 금액 양, 시간(3), 카테고리(2), 나이, 날(2), 12월에 경향성이 높음을 보여준다.  
