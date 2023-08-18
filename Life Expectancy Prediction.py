#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


#loading the data set
df = pd.read_csv('./Life Expectancy Data.csv')
df.head(10)


# In[3]:


df.shape


# In[ ]:





# In[4]:


#data cleaning
df.isnull().sum()


# In[5]:


#Replacing data with Medians
for x in list(df.columns): 
    if float(df['{}'.format(x)].isnull().sum()) != 0:
        df['{}'.format(x)].fillna((df['{}'.format(x)].median()), inplace = True)


# In[6]:


#No null values
df.isnull().sum()


# In[7]:


df1 = df.drop(['Year','Country', 'Status'], axis = 1)


# In[8]:


#distribution
for i in df1.columns:
    plt.figure()
    sns.distplot(df1[i])


# In[9]:


#for outliers
#skewed columns and Nearly Normal Distribution
skew = ['infant deaths', 'percentage expenditure', 'Hepatitis B', 'Measles ', 'under-five deaths ', 'Polio', 'GDP', 'Population',' HIV/AIDS', 'Diphtheria ', ' thinness  1-19 years', ' thinness 5-9 years']
norm = ['Life expectancy ', 'Adult Mortality', 'Alcohol', ' BMI ', 'Total expenditure', 'Income composition of resources', 'Schooling' ]


# In[10]:


#outliers analysis
df[skew].describe(percentiles=(1,0.99,0.9,0.75,0.5,0.3,0.1,0.01))
#Adultmortality, infant deaths, percentage expenditure, Measles, 


# In[11]:


#treating outliers
for x in skew:
    col_df = pd.DataFrame(df[x])
    col_median = col_df.median()
    Q3 = col_df.quantile(q=0.75)
    Q1 = col_df.quantile(q=0.25)
    IQR = Q3-Q1
    #deriving boundaries
    IQR_LL = float(Q1-1.5*IQR)
    IQR_UL = float(Q3+1.5*IQR)
    #Finding and treating outliers
    df.loc[df[x]>IQR_UL,x] = float(col_df.quantile(q=0.80))
    df.loc[df[x]<IQR_LL,x] = float(col_df.quantile(q=0.2))


# In[12]:


df[skew].describe(percentiles=(1,0.99,0.9,0.75,0.5,0.3,0.1,0.01))


# In[13]:


#removing leftover outliers
for x in ['Polio', ' HIV/AIDS', 'Diphtheria ']:
    col_df = pd.DataFrame(df[x])
    col_median = col_df.median()
    Q3 = col_df.quantile(q=0.75)
    Q1 = col_df.quantile(q=0.25)
    IQR = Q3-Q1
    #deriving boundaries
    IQR_LL = float(Q1-1.5*IQR)
    IQR_UL = float(Q3+1.5*IQR)
    #Finding and treating outliers
    df.loc[df[x]>IQR_UL,x] = float(col_df.quantile(q=0.80))
    df.loc[df[x]<IQR_LL,x] = float(col_df.quantile(q=0.2))


# In[14]:


fig, axes = plt.subplots(nrows=len(df[skew].columns), figsize=(8, 6 * len(df[skew].columns)))

for i, column in enumerate(df[skew].columns):
    axes[i].boxplot(df[column])
    axes[i].set_title(column)
    axes[i].set_ylabel('Values')

plt.tight_layout()
plt.show()


# In[ ]:





# In[15]:


#check the Outliers of norm
df[norm].describe(percentiles=(1,0.99,0.9,0.75,0.5,0.3,0.1,0.01))


# In[16]:


#outliers of Normal distributed columns
fig, axes = plt.subplots(nrows=len(df[norm].columns), figsize=(8, 6 * len(df[norm].columns)))

for i, column in enumerate(df[norm].columns):
    axes[i].boxplot(df[column])
    axes[i].set_title(column)
    axes[i].set_ylabel('Values')

plt.tight_layout()
plt.show()


# In[17]:


for x in norm:
    col_df = pd.DataFrame(df[x])
    col_median = col_df.median()
    Q3 = col_df.quantile(q=0.75)
    Q1 = col_df.quantile(q=0.25)
    IQR = Q3-Q1
    #deriving boundaries
    IQR_LL = float(Q1-1.5*IQR)
    IQR_UL = float(Q3+1.5*IQR)
    #Finding and treating outliers
    df.loc[df[x]>IQR_UL,x] = float(col_df.quantile(q=0.95))
    df.loc[df[x]<IQR_LL,x] = float(col_df.quantile(q=0.05))


# In[18]:


fig, axes = plt.subplots(nrows=len(df[norm].columns), figsize=(8, 6 * len(df[norm].columns)))

for i, column in enumerate(df[norm].columns):
    axes[i].boxplot(df[column])
    axes[i].set_title(column)
    axes[i].set_ylabel('Values')

plt.tight_layout()
plt.show()


# In[19]:


for x in ['Life expectancy ', 'Income composition of resources', 'Schooling']:
    col_df = pd.DataFrame(df[x])
    col_median = col_df.median()
    Q3 = col_df.quantile(q=0.75)
    Q1 = col_df.quantile(q=0.25)
    IQR = Q3-Q1
    #deriving boundaries
    IQR_LL = float(Q1-1.5*IQR)
    IQR_UL = float(Q3+1.5*IQR)
    #Finding and treating outliers
    df.loc[df[x]>IQR_UL,x] = float(col_df.quantile(q=0.75))
    df.loc[df[x]<IQR_LL,x] = float(col_df.quantile(q=0.25))


# In[20]:


for i in df.drop(['Country','Year','Status'], axis=1).columns:
    plt.figure()
    sns.distplot(df[i])


# In[21]:


def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
plt.figure(figsize=(14,14))
sns.heatmap(df.corr(),annot=True)


# In[22]:


cor_var = correlation(df,0.8)
cor_var


# In[23]:


####Linear Regression test without Scalling dependent Variable


# In[24]:


df2 = df.drop(['Year','Country','Status',' thinness 5-9 years', 'Diphtheria ', 'GDP', 'under-five deaths ', ' HIV/AIDS'], axis=1)
#define varibales
x = df2.drop('Life expectancy ',axis=1).to_numpy()
y = df2['Life expectancy '].to_numpy()
# Create Train Data set and Test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 62)

#scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)


# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
for name, method in [('Linear Regression', LinearRegression())]:
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
print('Method: {}'.format(name))
#coeffcients
print('\nIntercept: {:0.2f}'.format(float(method.intercept_)))
coeff_table=pd.DataFrame(np.transpose(method.coef_),df2.drop('Life expectancy ',axis=1).columns,columns=['Coefficients'])

print('\n')
print(coeff_table)

#MAE, MSE, RMSE
print('\nR2: {:0.2f}'.format(metrics.r2_score(y_test, predict)))
adjusted_r_squared2 = 1-(1-metrics.r2_score(y_test,predict))*(len(y)-1)/(len(y)-x.shape[1]-1)
print('Adj_R2: {:0.2f}'.format(adjusted_r_squared2))

print('\nMean Absolute Error: {:0.2f}'.format(metrics.mean_absolute_error(y_test, predict)))  
print('Mean Squared Error: {:0.2f}'.format(metrics.mean_squared_error(y_test, predict)))  
print('Root Mean Squared Error: {:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, predict)))) 


# In[26]:


#After scalling the dependent variable
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

df3 = df2
x2 = df3.drop('Life expectancy ',axis=1).to_numpy()
y2 = df3['Life expectancy '].to_numpy()
#split data
x_train3, x_test3, y_train3, y_test3 = train_test_split(x2,y2,test_size = 0.2, random_state=62)
# min max scaling on independent features
ms = MinMaxScaler()
x_train4 =ms.fit_transform(x_train3)
x_test4 = ms.transform(x_test3)
# scalling on Dependent variable
yj = PowerTransformer(method='yeo-johnson')
y_train4 = yj.fit_transform(y_train3.reshape(-1,1)).flatten()
y_test4 =yj.transform(y_test3.reshape(-1,1)).flatten()


# In[27]:


# create a model## delte
for name, method in [('Linear Regression', LinearRegression())]:
    method.fit(x_train4,y_train4)
    pred = method.predict(x_test4)

print('\nTransformed Model - Yeo-Johnson')
print('Method: {}'.format(name))
#coeffcients
print('\nIntercept: {:0.2f}'.format(float(method.intercept_)))
coeff_table=pd.DataFrame(np.transpose(method.coef_),df3.drop('Life expectancy ',axis=1).columns,columns=['Coefficients'])

print('\n')
print(coeff_table)

#MAE, MSE, RMSE
print('\nR2: {:0.2f}'.format(metrics.r2_score(y_test4, pred)))
adjusted_r_squared2 = 1-(1-metrics.r2_score(y_test4,pred))*(len(y2)-1)/(len(y2)-x2.shape[1]-1)
print('Adj_R2: {:0.2f}'.format(adjusted_r_squared2))

print('\nMean Absolute Error: {:0.2f}'.format(metrics.mean_absolute_error(y_test4, pred)))  
print('Mean Squared Error: {:0.2f}'.format(metrics.mean_squared_error(y_test4, pred)))  
print('Root Mean Squared Error: {:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test4, pred)))) 


# In[28]:


model = method.fit(x_train4,y_train4)
model.score(x_test4,y_test4)


# In[29]:


#feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def select_featuresCFS(x_train, y_train, x_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(x_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(x_train)
    # transform test input data
    X_test_fs = fs.transform(x_test)
    return X_train_fs, X_test_fs, fs

X_train_fs, X_test_fs, fs = select_featuresCFS(x_train4, y_train4, x_test4)
fs_table=pd.DataFrame(np.transpose(fs.scores_),df3.drop('Life expectancy ',axis=1).columns,
                          columns=['Feature Importance'])
print('Correlaton Feature Selection')
print(fs_table.sort_values(by=['Feature Importance'], ascending=False))


# In[30]:


#Feature Selection
#Eliminating least 4 important
df4 = df3.drop(['Population', 'Total expenditure', 'Hepatitis B', 'Measles '], axis=1)
x3 = df4.drop('Life expectancy ', axis =1).to_numpy()
y3 = df4['Life expectancy '].to_numpy()
#split data
x_train5, x_test5, y_train5, y_test5 = train_test_split(x3,y3,test_size = 0.2, random_state=62)
# min max scaling on independent features
ms = MinMaxScaler()
x_train6 =ms.fit_transform(x_train5)
x_test6 = ms.transform(x_test5)
# scalling on Dependent variable
yj = PowerTransformer(method='yeo-johnson')
y_train6 = yj.fit_transform(y_train5.reshape(-1,1)).flatten()
y_test6 =yj.transform(y_test5.reshape(-1,1)).flatten()

for name, method in [('Linear Regression', LinearRegression())]:
    method.fit(x_train6,y_train6)
    pred = method.predict(x_test6)

print('\nTransformed Model - Yeo-Johnson')
print('Method: {}'.format(name))
#coeffcients
print('\nIntercept: {:0.2f}'.format(float(method.intercept_)))
coeff_table=pd.DataFrame(np.transpose(method.coef_),df4.drop('Life expectancy ',axis=1).columns,columns=['Coefficients'])

print('\n')
print(coeff_table)

#MAE, MSE, RMSE
print('\nR2: {:0.2f}'.format(metrics.r2_score(y_test6, pred)))
adjusted_r_squared2 = 1-(1-metrics.r2_score(y_test6,pred))*(len(y3)-1)/(len(y3)-x2.shape[1]-1)
print('Adj_R2: {:0.2f}'.format(adjusted_r_squared2))

print('\nMean Absolute Error: {:0.2f}'.format(metrics.mean_absolute_error(y_test6, pred)))  
print('Mean Squared Error: {:0.2f}'.format(metrics.mean_squared_error(y_test6, pred)))  
print('Root Mean Squared Error: {:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test6, pred)))) 

 


# In[31]:


#tune hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
param = {'fit_intercept': [True, False],
    'normalize': [True, False]}
grid_search = GridSearchCV(model, param, cv = 10, scoring = 'neg_mean_squared_error')
grid_search.fit(x_train6,y_train6)

#best hyperparameters
print("best hyperparameters: ", grid_search.best_params_)
print("Best MSE: ", -grid_search.best_score_)


# In[32]:


#decision Tree
from sklearn.tree import DecisionTreeRegressor
for name,method in [('Decision Tree', DecisionTreeRegressor(random_state=62))]: 
    method.fit(x_train5,y_train5)
    pred = method.predict(x_test5)
print('Method: {}'.format(name))
#MAE, MSE, RMSE
print('\nR2: {:0.2f}'.format(metrics.r2_score(y_test5, pred)))
adjusted_r_squared2 = 1-(1-metrics.r2_score(y_test5,pred))*(len(y3)-1)/(len(y3)-x2.shape[1]-1)
print('Adj_R2: {:0.2f}'.format(adjusted_r_squared2))

print('\nMean Absolute Error: {:0.2f}'.format(metrics.mean_absolute_error(y_test5, pred)))  
print('Mean Squared Error: {:0.2f}'.format(metrics.mean_squared_error(y_test5, pred)))  
print('Root Mean Squared Error: {:0.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test5, pred)))) 


# In[33]:


# tune decisionn Tree #gid search
param1 = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10,12],
    'min_samples_leaf': [1, 3, 5]
}
dt_regressor = DecisionTreeRegressor()
grid_search = GridSearchCV(dt_regressor, param1, cv=12)
grid_search.fit(x_train5, y_train5)

# Print the best hyperparameters and best score
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

