--------------------------------------------------------
--  File created - Tuesday-September-26-2023   
--------------------------------------------------------
--------------------------------------------------------
--  DDL for Table FD
--------------------------------------------------------

  CREATE TABLE "FD" 
   (	"TRANS_NUM" VARCHAR2(128 BYTE) CONSTRAINT PK PRIMARY KEY, 
	"TRANS_DATE_TRANS_TIME" VARCHAR2(30 BYTE), 
	"CATEGORY" VARCHAR2(26 BYTE), 
	"AMT" FLOAT(126) NOT NULL, 
	"GENDER" VARCHAR2(26 BYTE), 
	"CITY" VARCHAR2(26 BYTE), 
	"STATE" VARCHAR2(26 BYTE), 
	"CITY_POP" NUMBER(38,0), 
	"JOB" VARCHAR2(128 BYTE), 
	"DOB" VARCHAR2(30 BYTE), 
	"IS_FRAUD" NUMBER(38,0) NOT NULL
   ) SEGMENT CREATION IMMEDIATE 
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "SYSTEM" ;

-- use this if import is not wroking due to ORA-01652 
-- alter database datafile 'C:\oraclexe\app\oracle\oradata\XE\SYSTEM.DBF' resize 3G;