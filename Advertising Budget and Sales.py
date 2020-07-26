#!/usr/bin/env python
# coding: utf-8

# In[8]:





# In[ ]:


import pandas as pd
import numpy as np


# In[2]:


advertising_data = pd.read_csv("Advertising Budget and Sales.csv")


# In[4]:


advertising_data.head()


# In[5]:


advertising_data.size


# In[7]:


advertising_data.shape


# In[11]:


x_features = advertising_data.iloc[:, 1:4]


# In[12]:


x_features.head()


# In[17]:


y_target = advertising_data.iloc[:, 4:5]


# In[19]:


y_target.head()


# In[21]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, random_state = 1, test_size = 0.25)


# In[24]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[26]:


from sklearn.linear_model import LinearRegression
lnr = LinearRegression()


# In[27]:


lnr.fit(x_train, y_train)


# In[28]:


print(lnr.intercept_)


# In[29]:


print(lnr.coef_)


# In[30]:


y_predict= lnr.predict(x_test)


# In[31]:


y_predict


# In[33]:


from sklearn.metrics import mean_squared_error


# In[34]:


MSE = (np.sqrt(mean_squared_error(y_test, y_predict)))


# In[35]:


MSE


# In[ ]:




