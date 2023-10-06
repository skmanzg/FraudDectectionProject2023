#!/usr/bin/env python
# coding: utf-8

# # Run this file after sql query (SQL 작업 후 시행합니다.)
# ## Data load from SQL query and do EDA and then export the file from pandas
# ## SQL작업 데이터 호출, EDA 후 저장
# install package if necessary 패키지 설치
get_ipython().system('pip install cx_Oracle')


import pandas as pd
import cx_Oracle as oci  # data import. it is told that python 3.4 is preferable.

# Go to C:\oraclexe\app\oracle\product\11.2.0\server\network\ADMIN. Then check 'port (1521)' and 'service name(xe)' in 'tnsname.ora'
# use your own information on user and pw.  (user와 pw변수는 각자 맞는 정보로 고쳐서 실행)
user = "DB"
pw = "1234"
dsn = "localhost:1521/xe" 


# connection 연결
con = oci.connect(user=user, password=pw, dsn=dsn)

# cursor 커서
cur = con.cursor()


query = 'select * from FD'
df = pd.read_sql_query(query, con)


# numeric labeling function for category, gender, state. (category, gender, state에 대한 numeric labeling 함수)
def column_labeling(DF, col_name):
    count = 0
    label_dict = {}
    column_list = []
    
    for i in DF[col_name]:
        if i in label_dict.keys():
            column_list.append(label_dict[i])
        else:
            label_dict[i] = count;
            count = count + 1
            column_list.append(label_dict[i])
            
    save_dict = pd.DataFrame()
    save_dict["string"] = label_dict.keys()
    save_dict["integer"] = label_dict.values()
    save_dict.to_csv("data/"+col_name+"_label.csv", index=False)
    
    DF[col_name] = column_list
    return DF

df = column_labeling(df, "CATEGORY")
df = column_labeling(df, "GENDER")
df = column_labeling(df, "STATE")


# ### New  Labeled Columns Index
# 
# __CATEGORY:__ 0 = misc_net, 1 = grocery_pos, 2 = entertainment, 3 = gas_transport, 4 = misc_pos, 5 = grocery_net, 6 = shooping_net, 7 = shopping_pos  
# 8 = food_dining, 9 = personal_care, 10 = health_fitness, 11 = travel, 12= kids_pets, 13 = home  
# * POS = point of sale, local payment while __net__ refers to 'online' payment
# 
# __GENDER:__ 0 = W, 1 = M  
# 
# __STATE:__ NC 0
# ,WA 1
# ,ID 2
# ,MT 3
# ,VA 4
# ,PA 5
# ,KS 6
# ,TN 7
# ,IA 8
# ,WV 9
# ,FL 10
# ,CA 11
# ,NM 12
# ,NJ 13
# ,OK 14
# ,IN 15
# ,MA 16
# ,TX 17
# ,WI 18
# ,MI 19
# ,WY 20
# ,HI 21
# ,NE 22
# ,OR 23
# ,LA 24
# ,DC 25
# ,KY 26
# ,NY 27
# ,MS 28
# ,UT 29
# ,AL 30
# ,AR 31
# ,MD 32
# ,GA 33
# ,ME 34
# ,AZ 35
# ,MN 36
# ,OH 37
# ,CO 38
# ,VT 39
# ,MO 40
# ,SC 41
# ,NV 42
# ,IL 43
# ,NH 44
# ,SD 45
# ,AK 46
# ,ND 47
# ,CT 48
# ,RI 49
# ,DE 50


# remove index
df = df.drop(columns='TRANS_NUM')  

# change the column sequence. 컬럼 순서 병경
df = df[['TRANS_YEAR','IS_COVID_YEAR', 'TRANS_MONTH','TRANS_DAY','TRANS_DAY_SIMPLIFIED','TRANS_HOUR','TRANS_HOUR_SIMPLIFIED','CATEGORY','AMT','GENDER','CITY','CITY_POP','STATE','JOB','AGE','IS_FRAUD', 'CC_NUM']]

# set binary data as bool
df['GENDER'] = df['GENDER'].astype('bool')
df['IS_COVID_YEAR'] = df['IS_COVID_YEAR'].astype('bool')
df['IS_FRAUD'] = df['IS_FRAUD'].astype('bool')

# put natural log on AMT column to improve
import numpy as np
df['log_AMT'] = np.log(df['AMT'])

# CITY_POP(BOX COX)
# box-cox can be easily applied through scipy.boxcox().  
# box cox 기법은 다음과 같은 optimal lambda를 구하고 적용을 시작한다.
from scipy import stats
df['BC_CITY_POP'], lambda_optimal = stats.boxcox(df['CITY_POP'])

# Save all process so far without index
# SQL 작업 + 지금까지의 분석 데이터 저장 (index 생성 방지 옵션 설정)
df.to_csv('data/Fraud_Detection_sql.csv', index=False)