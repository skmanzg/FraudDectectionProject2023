# FraudDetectionProject2023

1. Download fraudTest.csv and fraudTrain.csv from https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount

2. Execute Data_Wrangling_part1.ipynb before "!pip install cx_Oracle" code.

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

5. Execute SQL_work.sql

6. Execute SQL_datatype_modify.sql

7. Execute SQL_create_features.sql

8. Open Data_Wrangling_part1.ipynb again, and execute the rest codes.

9. Execute Data_Wrangling_part2.ipynb
(Data labeling)

10. Execute save_splited_table.ipynb

