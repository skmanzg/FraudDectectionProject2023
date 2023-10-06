# FraudDetectionProject2023

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