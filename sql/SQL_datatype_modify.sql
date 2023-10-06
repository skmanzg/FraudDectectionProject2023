-- DATE 형식의 VARCHAR2 컬럼2개 DATE로 바꾸기!
-- basic setting for 'MM/DD/YYYY HH24:MI' for two columns before use TO_DATE
ALTER SESSION SET NLS_DATE_FORMAT = "MM/DD/YYYY HH24:MI";

-- TEST to DOB (it also shows time but disregard it.)
SELECT DOB, TO_DATE(DOB,'MM/DD/YYYY') from fd;

-- APPLICATION to TABLE to create temp col(임시컬럼)
ALTER TABLE fd ADD (temp DATE);
UPDATE fd SET temp = DOB;

-- DROP and RENAME to have date type of DOB
ALTER TABLE fd DROP (DOB);
ALTER TABLE fd RENAME COLUMN temp TO DOB;

-- TEST to TRANS_DATE_TRANS_TIME (it works!)
SELECT TRANS_DATE_TRANS_TIME, TO_DATE(TRANS_DATE_TRANS_TIME,'MM/DD/YYYY HH24:MI') from fd;

-- APPLICATION to TABLE to create temp col(임시컬럼)
ALTER TABLE fd ADD (temp DATE);
UPDATE fd SET temp = TRANS_DATE_TRANS_TIME;

-- DROP and RENAME to have date type of TRANS_DATE_TRANS_TIME
ALTER TABLE fd DROP (TRANS_DATE_TRANS_TIME);
ALTER TABLE fd RENAME COLUMN temp TO TRANS_DATE_TRANS_TIME;

-- 커밋
commit;