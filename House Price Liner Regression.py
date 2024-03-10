#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# In[51]:


# Load the dataset
train_data = pd.read_csv("train.csv")  
test_data = pd.read_csv("test.csv")  



# In[52]:


train_data.head()


# In[53]:


test_data.head()


# In[54]:


# Selecting features and target variable
X_train = train_data.drop(columns=["SalePrice"])  # Features in the training set
y_train = train_data["SalePrice"]  # Target variable in the training set


# # **EDA (Exploratory Data Analysis)**
# 

# In[55]:


# Setting the max columns for pandas library

pd.set_option('display.max_columns',None)


# In[56]:


X_train.head()


# In[57]:


y_train.head()


# In[58]:


X_train.shape


# In[59]:


X_train.columns


# In[60]:


# Column info of train data
train_data.info()


# In[61]:


# Setting the max rows value to none for pandas
pd.set_option('display.max_rows',None)


# In[62]:


# Checking the number of null values in each column
X_train.isna().sum()


# In[63]:


# Dropping the columns with high NaN count

X_train = X_train.drop(['Alley','MasVnrType','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)


# In[64]:


# Fill missing values of LotFrontage with the mean
X_train.LotFrontage = X_train.LotFrontage.fillna(X_train.LotFrontage.mean())


# In[65]:


train_data = train_data.ffill()
test_data = test_data.ffill()


# In[66]:


#distribution and range of values in dataset columns
train_data.describe()


# In[67]:


# Again Checking the number of null values in each column
train_data.isna().sum()


# In[68]:


train_data = train_data.drop("Id",axis=1)


# In[69]:


train_data.head()


# In[70]:


# This is a list of column names that you want to drop for model training.

columns_to_encode = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                    'SaleType', 'SaleCondition']



train_data = train_data.drop(columns=columns_to_encode)
test_data = test_data.drop(columns=columns_to_encode)


# In[71]:


train_data.head()


# In[72]:


X=train_data.drop(columns='SalePrice')
y=train_data['SalePrice']


# In[73]:


X.head()


# In[74]:


y.head()


# In[75]:


# splits the dataset into training and testing sets, with 70% of the data used for training and 30% for testing, ensuring reproducibility with a random seed of 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[76]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[77]:


y_pred=model.predict(X_train)


# In[78]:


model.score(X_train,y_train)


# In[79]:


model.score(X_test,y_test)


# In[80]:


model.intercept_


# In[81]:


model.coef_


# In[82]:


SalePrice = model.predict(test_data.drop(columns='Id'))


# In[83]:


submission_df = pd.DataFrame({"Id": test_data['Id'], "SalePrice": SalePrice})


# In[84]:


submission_df.to_csv("LinearRegression.csv", index=False)


# In[85]:


data = pd.read_csv("LinearRegression.csv")
data.head()


# In[ ]:




