#!/usr/bin/env python
# coding: utf-8

#  SQL import 전, 원본 데이터 정리 및 병합
#  Data merging before SQL import

import pandas as pd

# Data load
fTrain = pd.read_csv('data/FraudTrain.csv')
fTest = pd.read_csv('data/FraudTest.csv')

# 'Unnamed:0' is nothing more than index, one unnecessary column, which needs to be dropped. 불필요 index 컬럼 삭제
fTD = fTest.drop(columns='Unnamed: 0')  

# Both data now can be merged into one via pd.concat(). 데이터 합치기
fDetection = pd.concat([fTrain, fTD], axis=0, join='inner')

# Save merged data without index. 병합 데이터 저장 (index 생성 방지)
fDetection.to_csv('data/Fraud_Detection.csv', index=False)  


