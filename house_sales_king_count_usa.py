#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install scikit-learn --upgrade --user')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# In[4]:


df.head()


# In[5]:


print(df.dtypes)


# In[6]:


df.describe()


# In[7]:


df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)


# In[8]:


df.describe()


# In[9]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[10]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[11]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[12]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[13]:


# Count the number of houses with unique floor values
floor_counts = df["floors"].value_counts().to_frame()

# Rename the column for better readability
floor_counts.columns = ["Number of Houses"]

# Display the result
print(floor_counts)


# In[14]:


# Create a boxplot for price based on waterfront view
plt.figure(figsize=(8, 6))  # Set figure size
sns.boxplot(x=df["waterfront"], y=df["price"])

# Add labels and title
plt.xlabel("Waterfront (0 = No, 1 = Yes)")
plt.ylabel("Price")
plt.title("Boxplot of Price by Waterfront View")
plt.show()


# In[15]:


# Create a regression plot
plt.figure(figsize=(8, 6))  # Set figure size
sns.regplot(x=df["sqft_above"], y=df["price"], scatter_kws={"alpha":0.5})

# Add labels and title
plt.xlabel("Square Foot Above Ground (sqft_above)")
plt.ylabel("Price")
plt.title("Regression Plot of Price vs. Sqft Above")
plt.show()


# In[17]:


df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns
print(df_numeric.corr()['price'].sort_values())  # Compute correlation safely


# In[18]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# In[19]:


# Define the feature (independent variable) and target (dependent variable)
X = df[['sqft_living']]  # Independent variable
Y = df['price']          # Dependent variable

# Create a Linear Regression model
lm = LinearRegression()

# Fit the model
lm.fit(X, Y)

# Calculate R² score (model performance)
r2_score = lm.score(X, Y)

# Print R² value
print(f"R² Score: {r2_score:.4f}")


# In[20]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     


# In[21]:


# Select features (independent variables) and target (dependent variable)
X = df[features]  # Independent variables
Y = df["price"]   # Target variable

# Create a Linear Regression model
lm = LinearRegression()

# Fit the model
lm.fit(X, Y)

# Calculate R² score
r2_score = lm.score(X, Y)

# Print R² value
print(f"R² Score: {r2_score:.4f}")


# In[22]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[23]:


# Define the list of features
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", 
            "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

# Select features (independent variables) and target (dependent variable)
X = df[features]  # Predictor variables
Y = df["price"]   # Target variable

# Create a pipeline object
pipeline = Pipeline([
    ('scale', StandardScaler()),               # Step 1: Standardize data
    ('polynomial', PolynomialFeatures(include_bias=False)),  # Step 2: Create polynomial features
    ('model', LinearRegression())              # Step 3: Apply Linear Regression
])

# Fit the pipeline
pipeline.fit(X, Y)

# Calculate R² score
r2_score = pipeline.score(X, Y)

# Print R² value
print(f"R² Score: {r2_score:.4f}")


# In[24]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[25]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[27]:


from sklearn.linear_model import Ridge
# Create and fit Ridge regression model with alpha=0.1
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)

# Calculate R² score on the test set
r2_score_test = ridge_model.score(x_test, y_test)

# Print R² value
print(f"R² Score on Test Data: {r2_score_test:.4f}")


# In[28]:


# Perform a second-order polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)  # Use transform (not fit_transform) on test data

# Create and fit Ridge regression model with alpha=0.1
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train_poly, y_train)

# Calculate R² score on the test set
r2_score_test = ridge_model.score(x_test_poly, y_test)

# Print R² value
print(f"R² Score on Test Data (Polynomial Ridge Regression): {r2_score_test:.4f}")


# In[ ]:




