-- Step 1: Create a Table in Oracle and Verify Data Import
SELECT * FROM telco_customer_churn FETCH FIRST 10 ROWS ONLY;


-- Step 2: Data Cleaning & Transformation
-- 2.1 Handle Null & Missing Values
-- Check for NULL values in each column:
SELECT column_name, COUNT(*) AS missing_values
FROM Telco_Customer_Churn
WHERE column_name IS NULL
GROUP BY column_name;

-- Fix TotalCharges Column (Potential Issue)
 -- Sometimes, TotalCharges is imported as a string due to blank values in CSV.
-- Convert it to numeric:

ALTER TABLE Telco_Customer_Churn ADD TotalCharges_numeric NUMBER;
UPDATE Telco_Customer_Churn SET TotalCharges_numeric = TO_NUMBER(NULLIF(TotalCharges, ''));

ALTER TABLE Telco_Customer_Churn DROP COLUMN TotalCharges;
RENAME TotalCharges_numeric TO TotalCharges;

-- 2.2 Handle Duplicates
-- Check for duplicate customerID:

SELECT customerID, COUNT(*)
FROM Telco_Customer_Churn
GROUP BY customerID
HAVING COUNT(*) > 1;

-- If duplicate Delete
DELETE FROM Telco_Customer_Churn
WHERE ROWID NOT IN (
    SELECT MIN(ROWID) FROM Telco_Customer_Churn GROUP BY customerID
);


/*
2.3 Encode Categorical Variables
Since ML models require numerical inputs, encode categorical variables:
*/

ALTER TABLE Telco_Customer_Churn ADD gender_encoded NUMBER;
UPDATE Telco_Customer_Churn SET gender_encoded = CASE WHEN gender = 'Male' THEN 1 ELSE 0 END;

-- Convert Multi-Class Categories (e.g., InternetService, Contract, PaymentMethod):

SELECT DISTINCT InternetService FROM Telco_Customer_Churn;


--Step 3: Exploratory Data Analysis (EDA)

--1. Count Total Customers
SELECT COUNT(*) FROM telco_customer_churn;

-- 2. Find Churn Rate
SELECT Churn, COUNT(*) AS Total_Customers 
FROM telco_customer_churn 
GROUP BY Churn;

-- 3. Find Average Monthly Charges by Contract Type
SELECT Contract, AVG(MonthlyCharges) AS Avg_Charges
FROM telco_customer_churn
GROUP BY Contract;

-- 4. Get Customers Who Churned
SELECT * FROM telco_customer_churn WHERE Churn = 'Yes';

-- 3.1 Summary Statistics
SELECT 
    MIN(tenure) AS Min_Tenure,
    MAX(tenure) AS Max_Tenure,
    AVG(tenure) AS Avg_Tenure,
    MIN(MonthlyCharges) AS Min_Charges,
    MAX(MonthlyCharges) AS Max_Charges,
    AVG(MonthlyCharges) AS Avg_Charges
FROM Telco_Customer_Churn;


-- 3.2 Churn Distribution
SELECT Churn, COUNT(*) AS Count, 
       ROUND((COUNT(*) * 100.0) / (SELECT COUNT(*) FROM Telco_Customer_Churn), 2) AS Percentage
FROM Telco_Customer_Churn
GROUP BY Churn;


-- 3.3 Customer Segmentation
SELECT Contract, Churn, COUNT(*) 
FROM Telco_Customer_Churn
GROUP BY Contract, Churn;

