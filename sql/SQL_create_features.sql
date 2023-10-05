-- Test AGE column
SELECT TRUNC(MONTHS_BETWEEN(TRUNC(SYSDATE), DOB)/12) AS AGE
FROM fd;

-- APPLICATION to TABLE to create AGE by sysdate.
-- the current time was 9/26/2023 EST.
ALTER TABLE fd ADD (AGE NUMBER);
UPDATE fd SET AGE = TRUNC(MONTHS_BETWEEN(TRUNC(SYSDATE), DOB)/12);

-- DROP DOB
ALTER TABLE fd DROP (DOB);


-- Test to see transaction Date and Time(hour only)
-- note that year has 2019 and 2020 only.
-- note that hour requires 'cast' to run
select TRANS_DATE_TRANS_TIME,
       extract (year from TRANS_DATE_TRANS_TIME) as Trans_year,
       extract (month from TRANS_DATE_TRANS_TIME) as Trans_month,
       extract (day from TRANS_DATE_TRANS_TIME) as Trans_day,
       extract (hour from cast(TRANS_DATE_TRANS_TIME as timestamp)) as Trans_hour
from fd;


-- add Trans_year
ALTER TABLE fd ADD (Trans_year NUMBER);
UPDATE fd SET Trans_year = extract (year from TRANS_DATE_TRANS_TIME);

-- add Trans_month
ALTER TABLE fd ADD (Trans_month NUMBER);
UPDATE fd SET Trans_month = extract (month from TRANS_DATE_TRANS_TIME);

-- add Trans_day
ALTER TABLE fd ADD (Trans_day NUMBER);
UPDATE fd SET Trans_day = extract (day from TRANS_DATE_TRANS_TIME);

-- add Trans_hour
ALTER TABLE fd ADD (Trans_hour NUMBER);
UPDATE fd SET Trans_hour = extract (hour from cast(TRANS_DATE_TRANS_TIME as timestamp));

commit


-- test hour split test  <- decode말고 case when 쓰자
select trans_hour, case when trans_hour between 0 and 6 Then 0
when trans_hour between 7 and 12 then 1
when trans_hour between 13 and 18 then 2 
else 3
End as Trans_hour_simplified
from fd;

-- update simplified hour
ALTER TABLE fd ADD (Trans_hour_simplified NUMBER);
UPDATE fd SET Trans_hour_simplified = case when trans_hour between 0 and 6 Then 0
when trans_hour between 7 and 12 then 1
when trans_hour between 13 and 18 then 2 
else 3
end;



-- test day split 
select trans_day, case when trans_day <= 10 Then 0
when trans_day between 11 and 20 then 1
when trans_day > 20 then 2 
End as Trans_day_simplified
from fd;

-- update simplified day
ALTER TABLE fd ADD (Trans_day_simplified NUMBER);
UPDATE fd SET Trans_day_simplified =case when trans_day <= 10 Then 0
when trans_day between 11 and 20 then 1
when trans_day > 20 then 2 
End;


-- test year split
select trans_year, case when trans_year = 2019 Then 0
when trans_year = 2020 then 1
End as IS_COVID_YEAR
from fd;

-- update IS_COVID_YEAR
ALTER TABLE fd ADD (IS_COVID_YEAR NUMBER);
UPDATE fd SET IS_COVID_YEAR = case when trans_year = 2019 Then 0
when trans_year = 2020 then 1
End;


-- 불필요 col 삭제
ALTER TABLE fd DROP (TRANS_DATE_TRANS_TIME);


commit