# Python-loan-defaulter
Create a model to predict the potential defaulters

 **Domain**: Finance

 **Project Name:** Determine and examine factors that affected the ratio of vehicle loan defaulters. Also, use the findings to create a model to predict the potential defaulters.

 **Problem:** Financial institutions incur significant losses due to the default of vehicle loans. This has led to the tightening up of vehicle loan underwriting and increased vehicle loan rejection rates.
The need for a better credit risk scoring model is also raised by these institutions. This warrants a study to estimate the determinants of vehicle loan default.
There is 1 dataset data that have 41 attributes.
You are required to determine and examine factors that affected the ratio of vehicle loan defaulters. Also, use the findings to create a model to predict the potential defaulters.

## **Performed Tasks:**
 - Load package for mathematical calculations.
 - Use csv file Loandata.csv as a data source
 - EDA (Exploratory data analysis)
   
   •	loan.head  To see 5 top records of data
   
   •	loan.shape To see no of columns and rows
   
   •	loan.describe() To see statistical values of quantitative variable
   
   •	loan.columns To check variable names
   
   •	loan.info() To Check basic info, data types
   
   •	loan.isna().sum() To check count of null values
   
   •	loan.duplicated().sum() To Check duplicate value

## EDA findings:
   •	Some of the variable names are not as per python variable naming conventions.
   
   •	Employment Type has 7661 null values.
## Solution of above findings:
   •	. from variable names are replaced with _.
   
   •	As Employment type is categorical variable so null values should be replaced by mode function.

## Univariate Analysis
***********************
In this section analyzing distribution of target variable "loan_default".
    - To plot graphs loading packages for graph plotting(matplotlib.pyplot) and data visualization(seaborn).
    - Plotting bar graph for target variable and check the distribution.
    - Find out no of defaulter and non-defaulter from loan_default target variable.
    - Find out % of defaulter and non-defaulter from loan_default target variable.
## Findings:
    - 78.29% of customers are non-defaulters and 21.7% are defaulter.
    
## Bivariate Analysis
***********************
In this section analyzing distribution of independent variables with target variable.
Considering independent variables are branch_id, State_ID, manufacturer_id, supplier_id, Employment_Type, Age, PERFORM_CNS_SCORE, NO_OF_INQUIRIES, SANCTIONED_AMOUNT, DISBURSED_AMOUNT.
## Findings:
All above variables does not affect much to decide defaulter or non-defaulter.

## Model creation for potential defaulters prediction
******************************************************
 - Load different packages to create model like below:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

 - Make a copy of dataset so that there is no mugging up in original dataset.
 - Drop irrelevant columns from copied dataset inp1.
 - Divide inp1 into two parts, first(y) will hold target variable loan_default only and other(X) will hold rest of the variables.
 - Standardized the later part of dataset.
 - Split(80:20) both of the data parts into train and test data sets respectively.
 - Make logistic regression model, fit train dataset into the model and predict model with test data.
 - Find accuracy score with the help of actual and predicted data which comes as 78.68%
 - confusion matrix is as
     array([[36653,    80],
   
           [ 9858,    40]], dtype=int64)

![alt text](https://github.com/AnubhaRanjan/Python-loan-defaulter/blob/main/confusion%20matrix%20pic.png)


TP= 36653

FN=80

FP=9858

TN=40

**Accuracy** = (TP+TN)/(TP+FP+FN+TN)= 78.6%

**Precision** = TP/(TP+FP) = 78.8%

**Note:** Data file and code file are attached. Data file is in zipped format.

**Recall/Sensitivity** = TP/(TP+FN) = 99.77%

**F1 Score** = 2(p*r)/(p+r) where ‘p’ is precision and ‘r’ is recall= 0.88

