#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#-------------------California Housing society project

Background of Problem Statement :

The US Census Bureau has published California Census Data which has 10 types of metrics such as the population, median income, median housing price, and so on for each block group in California. The dataset also serves as an input for project scoping and tries to specify the functional and nonfunctional requirements for it.

Problem Objective :

The project aims at building a model of housing prices to predict median house values in California using the provided dataset. This model should learn from the data and be able to predict the median housing price in any district, given all the other metrics.

Districts or block groups are the smallest geographical units for which the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). There are 20,640 districts in the project dataset.

Analysis Tasks to be performed:

Build a model of housing prices to predict median house values in California using the provided dataset.

Train the model to learn from the data to predict the median housing price in any district, given all the other metrics.

Predict housing prices based on median_income and plot the regression chart for it....


# In[ ]:


#Importing numpy and Pandas libraries


# In[173]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


#Reading data from excel


# In[174]:


housing_data = pd.read_excel("1553768847_housing.xlsx")


# In[175]:


housing_data.fillna(housing_data.mean(), inplace = True)


# In[167]:


housing_data.dtypes


# In[ ]:


#Finding Columns which has data type object


# In[59]:


housing_data.select_dtypes(include = ['object'])


# In[ ]:


#Convert Object data type to numerical


# In[67]:


housing_data['ocean_proximity_code'] = pd.factorize(housing_data.ocean_proximity)[0]


# In[61]:


housing_data.ocean_proximity.value_counts()


# In[ ]:


#Method 2


# In[168]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
housing_data['ocean_proximity_code'] = lb.fit_transform(housing_data.ocean_proximity)


# In[133]:


#Method 3


# In[176]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
housing_data['ocean_proximity'] = lb.fit_transform(housing_data['ocean_proximity'].astype('str'))


# In[109]:


columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity_code','median_house_value', 'ocean_proximity']


# In[110]:


housing = housing_data.reindex(columns = columns)


# In[147]:


housing.head()


# In[41]:


#Define Input and Output Data


# In[177]:


x_input = housing_data.iloc[:, 0:9]
y_output = housing_data.iloc[:, 9:10]


# In[ ]:


#Split Train and Test Data Sets


# In[178]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size = 0.2, random_state = 50)


# In[179]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[157]:


x_train


# In[180]:


#Standardise
from sklearn import preprocessing
scalar = preprocessing.StandardScaler()
scalar.fit(x_train)
x_train_new = scalar.transform(x_train)
x_test_new =scalar.transform(x_test)


# In[188]:


#Fit model to dataset
from sklearn.linear_model import LinearRegression
lnr=LinearRegression()
lnr.fit(x_train_new, y_train)


# In[189]:


y_test_pred=lnr.predict(x_test_new)


# In[190]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
rmse


# In[202]:


#Predict housing prices based on median_income and plot the regression chart for it....
x_test_median_income_alone = x_test.iloc[:, 7:8]
x_test_median_income_alone
x_train_median_income = x_train.iloc[:, 7:8]
x_train_median_income


# In[204]:


x_train_median_income.shape


# In[205]:


fit_model = lnr.fit(x_train_median_income,y_train)


# In[207]:


y_test_median_income = lnr.predict(x_test_median_income_alone)


# In[209]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_test_median_income))
rmse


# In[220]:


import matplotlib.pyplot as plt
plt.scatter(x_test_median_income_alone, y_test_median_income, color='green')
plt.plot(x_test_median_income_alone, y_test_median_income, color = 'red')
plt.show()


# In[219]:


plt.scatter(x_train_median_income, y_train)
plt.plot(x_train_median_income, y_train, color = 'red')
plt.show()


# In[ ]:




