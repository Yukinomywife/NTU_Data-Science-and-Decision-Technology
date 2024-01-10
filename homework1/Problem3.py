# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:11:56 2023

@author: jimmy
"""
import numpy as np
import pandas as pd
import pandas.io.formats.excel
import statsmodels.api as sm
from sklearn.model_selection import train_test_split  #(c)
pandas.io.formats.excel.ExcelFormatter.header_style = None
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

f = pd.read_excel("Concrete-Homework.xlsx")
df = f.drop(['Unnamed: 0'], axis=1)

#model chosen: 'LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor'
model = 'LinearRegression'

df = df.dropna()
df = df.reset_index(drop = True)

if model == 'LinearRegression':
    df['Water'] = np.log(df['Water'])
    df['CoarseAggregate'] = np.log(df['CoarseAggregate'])
    df['FineAggregate'] = np.power(df['FineAggregate'], -1.014)
    df['Age'] = np.power(df['Age'], 0.2754)
else:
    df = df

scaler = MinMaxScaler()
df[df.columns[:]] = scaler.fit_transform(df[df.columns[:]])

new_feature = df['Age'] ** 1.5 
df.insert(len(df.columns)-1, column = "Age^1.5", value = new_feature)

train, test = train_test_split(df,test_size = 0.2,random_state = 4172888)

df1 = pd.DataFrame(train, columns = df.columns)
df2 = pd.DataFrame(test, columns = df.columns)

#Regression
y = df1.iloc[:,len(df1.columns)-1]
X = df1.iloc[:,:len(df1.columns)-1]
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
#print(est2.summary(),'\n')

#model changed
if model == 'LinearRegression':
    model = LinearRegression()
if model == 'DecisionTreeRegressor':
    model = DecisionTreeRegressor(random_state=42)
if model == 'RandomForestRegressor':
    model = RandomForestRegressor(random_state=925)
model.fit(X, y)

X_test = df2.iloc[:,:len(df2.columns)-1]
y_test = df2.iloc[:,len(df2.columns)-1]
predictions_train = model.predict(X)
predictions_test = model.predict(X_test)
print('R-square_train:',model.score(X, y))
print('RMSE_train:',mean_squared_error(y,predictions_train,squared = False),'\n')

print('R-square_test:',model.score(X_test, y_test))
print('RMSE_test:',mean_squared_error(y_test,predictions_test,squared = False))

df1.to_excel("Concrete-Homework-train.xlsx",index = False)
df2.to_excel("Concrete-Homework-test.xlsx",index = False)

n = X_test.shape[1]
adjusted_r_squared = 1 - (1 - model.score(X_test, y_test)) * ((len(y_test) - 1) / (len(y_test) - n - 1))

print("Adjusted R-squared:", adjusted_r_squared)
