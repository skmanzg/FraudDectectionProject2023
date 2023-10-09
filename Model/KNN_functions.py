#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from tensorflow import keras


# In[4]:


def ColumnsExtractor(data, isLearn):
    reduced_data = pd.DataFrame([], columns=["AMT", "TRANS_MONTH", "TRANS_DAY", "TRANS_HOUR", "CATEGORY", "IS_FRAUD"])
    reduced_data = data[["AMT", "TRANS_MONTH", "TRANS_DAY", "TRANS_HOUR", "CATEGORY", "IS_FRAUD"]]
    if(isLearn): # 데이터는 여러 행
        reduced_data.loc[:,"IS_FRAUD"] = reduced_data.loc[:,"IS_FRAUD"].astype("int")
    else: # 데이터는 1개의 행
        reduced_data["IS_FRAUD"] = reduced_data["IS_FRAUD"].astype("int")
    reduced_data.to_csv("reduced_data.csv",index=False)
    return reduced_data


# In[5]:


def KNNforFraud(df, isLearn, beta, delta1, epsilon):
    if(not isLearn):
        #학습이 아닌 추론일 때만 사용됨
        df_train_x = pd.read_csv("../data/splitted/Fraud_Detection_train_features.csv")
        df_train_y = pd.read_csv("../data/splitted/Fraud_Detection_train_target.csv")
        df_train = pd.concat([df_train_x, df_train_y], axis=1)
        df_train = ColumnsExtractor(df_train, not isLearn)
    
    reduced_data = ColumnsExtractor(df, isLearn)
    
    #스케일링 함수로 객체만들고 scaled된 reduced_data_scaled 만들기
    MMscaler = RobustScaler()
    if(isLearn):
        MMscaler.fit(reduced_data)
        reduced_data_scaled = MMscaler.transform(reduced_data)
        reduced_data_scaled = pd.DataFrame(reduced_data_scaled, columns=["AMT", "TRANS_MONTH", "TRANS_DAY", "TRANS_HOUR", "CATEGORY", "IS_FRAUD"])
    else:
        #학습용 데이터 맨 아래에 임시로 행을 붙인 뒤 다시 뗀다
        df_temp = pd.concat([df_train, reduced_data], axis = 0)
        MMscaler.fit(df_temp)
        df_temp2 = MMscaler.transform(df_temp)
        df_temp2 = pd.DataFrame(df_temp2, columns=["AMT", "TRANS_MONTH", "TRANS_DAY", "TRANS_HOUR", "CATEGORY", "IS_FRAUD"])
        reduced_data_scaled = df_temp2.tail(1)

    #5개의 KNN 객체 생성. parameter는 반드시 odd number로!
    global kn1
    global kn2
    global kn3
    global kn4
    global kn5
    
    global FCmodel
    
    if(isLearn):
        kn1 = KNeighborsClassifier(n_neighbors=3)
        kn2 = KNeighborsClassifier(n_neighbors=3)
        kn3 = KNeighborsClassifier(n_neighbors=3)
        kn4 = KNeighborsClassifier(n_neighbors=3)
        kn5 = KNeighborsClassifier(n_neighbors=3)
        
        kn1.fit(reduced_data_scaled[["AMT", "TRANS_MONTH"]], reduced_data_scaled["IS_FRAUD"])
        kn2.fit(reduced_data_scaled[["TRANS_MONTH", "TRANS_DAY"]], reduced_data_scaled["IS_FRAUD"])
        kn3.fit(reduced_data_scaled[["AMT", "TRANS_DAY"]], reduced_data_scaled["IS_FRAUD"])
        kn4.fit(reduced_data_scaled[["AMT", "TRANS_HOUR"]], reduced_data_scaled["IS_FRAUD"])
        kn5.fit(reduced_data_scaled[["AMT", "CATEGORY"]], reduced_data_scaled["IS_FRAUD"])
        arr1 = kn1.predict_proba(reduced_data_scaled[["AMT", "TRANS_MONTH"]])
        arr2 = kn2.predict_proba(reduced_data_scaled[["TRANS_MONTH", "TRANS_DAY"]])
        arr3 = kn3.predict_proba(reduced_data_scaled[["AMT", "TRANS_DAY"]])
        arr4 = kn4.predict_proba(reduced_data_scaled[["AMT", "TRANS_HOUR"]])
        arr5 = kn5.predict_proba(reduced_data_scaled[["AMT", "CATEGORY"]])
        
        arr1 = arr1.reshape(-1, 2)
        arr2 = arr2.reshape(-1, 2)
        arr3 = arr3.reshape(-1, 2)
        arr4 = arr4.reshape(-1, 2)
        arr5 = arr5.reshape(-1, 2)
        
        #arr1~5를 하나의 numpy 배열로 합칠 것이다. 하나의 데이터는 하나의 행이다.
        NNinput = arr1
        NNinput = np.append(NNinput, arr2, axis=1)
        NNinput = np.append(NNinput, arr3, axis=1)
        NNinput = np.append(NNinput, arr4, axis=1)
        NNinput = np.append(NNinput, arr5, axis=1)
        NNinput = np.delete(NNinput, 0, axis=1)
        NNinput = np.delete(NNinput, 1, axis=1)
        NNinput = np.delete(NNinput, 2, axis=1)
        NNinput = np.delete(NNinput, 3, axis=1)
        NNinput = np.delete(NNinput, 4, axis=1)
        
        NNoutput = np.array(reduced_data_scaled["IS_FRAUD"].to_list())
        NNoutput = NNoutput.T
        
        #Fully Connected층을 epoch = epsilon횟수로 학습
        FCmodel = keras.Sequential()
        FCmodel.add(keras.layers.Dense(beta, input_shape = (5, )))
        FCmodel.add(keras.layers.Dense(1, activation="sigmoid"))
        FCmodel.compile(loss="binary_crossentropy", metrics="accuracy")
        FCmodel.fit(NNinput, NNoutput, epochs=epsilon)
        return 2
    else:
        arr1 = kn1.predict_proba(reduced_data_scaled[["AMT", "TRANS_MONTH"]])
        arr2 = kn2.predict_proba(reduced_data_scaled[["TRANS_MONTH", "TRANS_DAY"]])
        arr3 = kn3.predict_proba(reduced_data_scaled[["AMT", "TRANS_DAY"]])
        arr4 = kn4.predict_proba(reduced_data_scaled[["AMT", "TRANS_HOUR"]])
        arr5 = kn5.predict_proba(reduced_data_scaled[["AMT", "CATEGORY"]])
        
        arr1 = arr1[:,0]
        arr2 = arr2[:,0]
        arr3 = arr3[:,0]
        arr4 = arr4[:,0]
        arr5 = arr5[:,0]
        
        arr_input = arr1
        arr_input = np.append(arr_input, arr2)
        arr_input = np.append(arr_input, arr3)
        arr_input = np.append(arr_input, arr4)
        arr_input = np.append(arr_input, arr5)
        
        arr_input_df = pd.DataFrame(arr_input)

        arr_input_df = arr_input_df.T
        
        #print("FCmodel will be Executed...\n")
        fraud_predict = FCmodel.predict(arr_input_df)
        fraud_predict = float(fraud_predict)
        
        if(fraud_predict > delta1):
            return 1
        else:
            return 0

