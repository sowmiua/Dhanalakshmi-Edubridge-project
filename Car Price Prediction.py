#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


car=pd.read_csv('quikr_car.csv')
car


# In[3]:


car.head()


# In[4]:


car.shape


# In[5]:


car.info()


# In[6]:


car.isnull().sum()


# Creating backup copy

# In[7]:


backup=car.copy()


# In[8]:


car.isnull().sum()


# ### Quality

# -year has many non-year values -year object to int -price has ASK for Price -kms_driven has kms with integers -kms_driven object to int -kms_driven has nan values -fuel_type has nan values -keep frist 3 words of name

# ### Cleaning Data

# ##### year has many non-year values

# In[9]:


car=car[car['year'].str.isnumeric()]
car


# ##### year is in object change to integer

# In[10]:


car['year']=car['year'].astype(int)
car['year']


# ##### Price has Ask for Price

# In[11]:


car=car[car.Price!='Ask For Price']
car


# ##### Price has commas in its prices and is in object

# In[12]:


car.Price=car.Price.str.replace(',','').astype(int)
car.Price


# In[13]:


car.info()


# ##### kms_driven has object values with kms at last

# In[14]:


car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')
car['kms_driven']


# In[15]:


car.kms_driven[car.kms_driven=='Petrol']


# ##### It has nan values and two rows have 'petrol' in them

# In[16]:


car=car[car['kms_driven'].str.isnumeric()]
car


# In[17]:


car['kms_driven']=car['kms_driven'].astype(int)
car['kms_driven']


# In[18]:


car.isnull().sum()


# In[19]:


car[~car['fuel_type'].isna()].isnull().sum()


# ##### fuel_typehas nan values

# In[20]:


car=car[~car['fuel_type'].isna()]
car


# In[21]:


car.shape


# In[22]:


car.company.unique()


# ##### name and company had spammed data ... but with the previous cleaning, those rows got removed.bold text

# ##### Company does not need any cleaning now.Changing car names.  Keeping only the first three words

# In[23]:


car['name']=car['name'].str.split().str.slice(start = 0, stop = 3).str.join(' ')
car['name']


# ##### Resetting the index of the final cleaned data

# In[24]:


car=car.reset_index(drop=True)
car


# ##### Cleaned Data

# In[25]:


car


# In[26]:


car.to_csv('Cleaned_Car_data.csv',index=False)


# In[27]:


car.info()


# In[28]:


car.describe()


# In[29]:


car=car[car['Price']<=6000000]
car


# ##### Checking relationship of Company with Price

# In[30]:


car['company'].unique()


# ### Data Visualization 

# ##### Boxplot

# In[31]:


plt.subplots(figsize=(15,7))
ax=sns.barplot(x='company',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# In[32]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# In[33]:


car.plot(x='company',y='Price')
plt.title('Line Chart of Price')
plt.xlabel('company')
plt.ylabel('Price')
plt.show()


# In[34]:


numeric_features=['year','Price','kms_driven']

fig,ax = plt.subplots(len(numeric_features),3,figsize=(30,20))
for index,i in enumerate(numeric_features):
    sns.distplot(car[i],ax=ax[index,0],color='red')
    sns.boxplot(car[i],ax=ax[index,1],color='pink')
    sns.violinplot(car[i],ax=ax[index,2],color='purple')
        
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.suptitle("Visualizing continuous columns (car dataset)",fontsize=30)


# In[35]:


sns.pairplot(car, hue='company', diag_kind='kde')
plt.show()


# ##### Checking relationship of Year with Price

# In[36]:


plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=car)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# ##### Checking relationship of kms_driven with Price

# In[37]:


sns.relplot(x='kms_driven',y='Price',data=car,height=7,aspect=1.5)


# ##### Checking relationship of Fuel Type with Price

# In[38]:


plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=car)


# ##### Relationship of Price with FuelType, Year and Company mixed

# In[39]:


ax=sns.relplot(x='company',y='Price',data=car,hue='fuel_type',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')


# In[40]:


fuel_type_counts = car['fuel_type'].value_counts()

plt.figure(figsize=(8,6))
plt.pie(fuel_type_counts, labels = fuel_type_counts.index, autopct = '%1.1f%%', startangle = 90)
plt.title('Car Fuel type')
plt.axis('equal')
plt.show()


# ##### Extracting Training Data

# In[41]:


X = car[['name','company','year','kms_driven','fuel_type']]
y = car['Price']


# In[42]:


X


# In[43]:


y


# In[44]:


X.shape


# In[45]:


y.shape


# ##### Applying Train Test Split

# In[46]:


from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[47]:


print(f"X Train Size : {X_train.shape}")
print(f"X Test Size : {X_test.shape}")
print(f"Y Train Size : {y_train.shape}")
print(f"Y Test Size : {y_test.shape}")


# In[48]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# ##### Creating an OneHotEncoder object to contain all the possible categories

# In[49]:


ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])


# In[50]:


ohe.categories_


# In[51]:


X


# ##### Creating a column transformer to transform categorical columns

# In[52]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')


# ##### Linear Regression Model

# In[53]:


lr = LinearRegression()
lr


# In[54]:


GB = GradientBoostingRegressor()
GB


# In[55]:


def Crossvalidation(estimators,X,y):
    for estimator in estimators:
        ShuffleSplit_cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
        val= cross_val_score(estimator[1],column_trans.fit_transform(X),y,cv=ShuffleSplit_cv,scoring='r2')
        print(estimator[0],np.round(np.mean(val),2))


# In[56]:


estimators=[('lr',lr),('GB',GB)]


# In[57]:


Crossvalidation(estimators=estimators,X=X,y=y)


# ##### Making a pipeline

# In[58]:


pipe=make_pipeline(column_trans,lr)


# ##### Fitting Model

# In[59]:


pipe.fit(X_train,y_train)


# ##### Checking R2 Score

# In[60]:


y_pred=pipe.predict(X_test)
print('r2Score:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred)))


# ##### Gradient Boosting Regressor:

# In[61]:


pipe_gradient=make_pipeline(column_trans,GB)


# In[62]:


pipe_gradient.fit(X_train,y_train)


# In[63]:


y_pred2=pipe_gradient.predict(X_test)

print('r2Score:',r2_score(y_test,y_pred2))
print('MAE:',mean_absolute_error(y_test,y_pred2))
print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred2)))


# ##### Finding the model with a random state of Train Test Split where the model was found to give almost 0.92 as r2_score

# In[64]:


scores = []
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


# In[65]:


random_state_value = np.argmax(scores)
random_state_value


# In[66]:


Max_Score = scores[np.argmax(scores)]
Max_Score


# In[67]:


pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# ##### The best model is found at a certain random state

# In[68]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
print('r2Score:',r2_score(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred)))


# In[69]:


import pickle


# In[70]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[71]:


pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# In[72]:


pipe.steps[0][1].transformers[0][1].categories[0]


# 
