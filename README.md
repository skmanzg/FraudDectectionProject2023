# FraudDetectionProject2023

## 팀원명
* 나도엽
* 신승운

## 데이터셋 출처
https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount

## 프로젝트 개요
* 이 프로젝트는 신용카드의 부정거래 데이터를 분석하고, 패턴을 파악하여 그 인사이트를 기반으로 모델을 설계, 구현하는 프로젝트입니다.
* 이 프로젝트로부터 나온 인공지능은 80%의 정확도로 신용카드 부정거래 여부를 탐지합니다.

## 프로젝트 진행 순서
1. kaggle에서 데이터 수집
2. Oracle SQL로 데이터 가공
3. 데이터 분석
4. 인공지능 아키텍처 설계
5. Python 및 파이썬 라이브러리를 통해 인공지능 모델 구현
6. 모델 테스트

### 데이터 수집
다음의 웹 사이트에서 fraudTest.csv와 fraudTrain.csv를 다운로드
https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount

### 데이터 가공
* Oracle SQL을 기반으로 하며, Oracle SQL Developer 툴을 사용함

사용된 컬럼들 정보
	* TRANS_NUM
	* CC_NUM
	* TRANS_DATE_TRANS_TIME
	* CATEGORY
	* AMT
	* GENDER
	* CITY
	* STATE
	* CITY_POP
	* JOB
	* DOB
	* IS_FRAUD

IS_FRAUD는 TARGET 정보임

날짜 형식이 데이터마다 제각각이기 때문에 날짜 형식을 통일하는 SQL 쿼리를 대입

### 데이터 분석
가공된 데이터로부터 패턴을 추출

(사진)
하나의 CC_NUM에 대해서 시간 순으로 정렬했을 때, 거래량 차이가 크면 FRAUD일 가능성이 크다고 판단함

### 인공지능 아키텍처 설계
(사진)
(사진)

### 인공지능 모델 구현
모델은 py확장자를 사용하여 ipynb에서 import하는 식으로 구성


### 모델 테스트
테스트 세트로 검증 결과 정확도 80%를 보임

-------------------------

## 따라하는 방법

1. Download fraudTest.csv and fraudTrain.csv from https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount

2. Run Data_Wrangling_part1.ipynb 

3. Open Oracle SQL Developer and execute Table_script.sql

format of table's columns is,
TRANS_NUM
CC_NUM
TRANS_DATE_TRANS_TIME
CATEGORY
AMT
GENDER
CITY
STATE
CITY_POP
JOB
DOB
IS_FRAUD

4. Import data to "FD" table from data/Fraud_Detection.csv
(Check the tablespace storage size)

If tablespace is lack, type the sql prompt command.
a. SELECT * FROM DBA_DATA_FILES;
-> Find the path of SYSTEM.DBF
b. ALTER DATABASE DATAFILE 'path' RESIZE 9G;

5. Run SQL_work.sql

6. Run SQL_datatype_modify.sql

7. Run SQL_create_features.sql

8. Run Data_Wrangling_part2.ipynb

9. Run save_splitted_table.ipynb

10. Run data_analysis.ipynb and data_analysis2.ipynb to see the graph.

11. Run Model/Model_main.ipynb for learning and testing the model.