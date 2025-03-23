# house_sales_analysis
This repository contains a Data Analysis and Machine Learning project focused on predicting house prices using Python. The project involves data cleaning, exploratory data analysis (EDA), visualization, and predictive modeling using linear regression and ridge regression.

DATA WRANGLING<br/>
Dropped the columns "id"  and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data<br/>
<br/>
EXPLORATORY DATA ANALYSIS<br/>
Used the method value_counts to count the  number of houses with unique floor values and also used the method .to_frame() to convert it to a dataframe<br/>
Used the function boxplot in the seaborn library to determine whether houses with a waterfront or without a waterfront view have more price outliers<br/>
Used the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price<br/>
<br/>
MODEL DEVELOPMENT<br/>
Model Evaluation and Refinement<br/>

