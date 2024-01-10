# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 21:54:32 2023

@author: jimmy
"""
import numpy as np
import pandas as pd
import pandas.io.formats.excel
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
pandas.io.formats.excel.ExcelFormatter.header_style = None
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("Problem 2: The fish dataset (Fish-3Only-Homework) contains the weight, height, and width data of 3 different kinds of fish: Bream, Roach, and Perch")
df = pd.read_excel("Fish Weight-3Only.xlsx")

#a. delete missing data
values = [0]
df = df[df.isin(values) == False]
df = df.dropna()
df = df.reset_index(drop = True)

x0 = []
x1 = []
x2 = []

#b.
for i in range(len(df)):
    if df.Species[i] == 'Bream':
        x0.append(int(1))
        x1.append(int(0))
        x2.append(int(0))
    if df.Species[i] == 'Roach':
        x0.append(int(0))
        x1.append(int(1))
        x2.append(int(0))    
    if df.Species[i] == 'Perch':
        x0.append(int(0))
        x1.append(int(0))
        x2.append(int(1))

df.insert(7, column = "Bream-D", value = x0)
df.insert(8, column = "Roach-D", value = x1)
df.insert(9, column = "Perch-D", value = x2)

ii = 0
count = 0
count1 = 0

while ii < len(df) :
    if df.Species[ii] == 'Bream':
        count += 1
        count1 += 1
        B = df.iloc[:ii+1,:].values
    if df.Species[ii] == 'Roach':
        count1 += 1
        R = df.iloc[count:ii+1,:].values
    if df.Species[ii] == 'Perch':
        P = df.iloc[count1:ii+1,:].values
    ii += 1

#c.
B_train, B_test = train_test_split(B,test_size = 0.2,random_state = 88)
R_train,R_test = train_test_split(R,test_size = 0.2,random_state = 88)
P_train,P_test = train_test_split(P,test_size = 0.2,random_state = 88)

train = np.concatenate((B_train, R_train))
train = np.concatenate((train, P_train))
test = np.concatenate((B_test, R_test))
test = np.concatenate((test, P_test))

df1 = pd.DataFrame(train, columns = df.columns)
df2 = pd.DataFrame(test, columns = df.columns)

Weight = df1.iloc[:,1]
X = df1.iloc[:,2:]
#d.
X2 = sm.add_constant(X)
est = sm.OLS(Weight.astype(float), X2.astype(float))
est2 = est.fit()
print(est2.summary(),'\n')

model = LinearRegression()
model.fit(X, Weight)
X_test = df2.iloc[:,2:]
y_test = df2.iloc[:,1]
predictions_train = model.predict(X)
predictions_test = model.predict(X_test)
print('R-square of the training set: ',model.score(X, Weight))
print('RMSE of the training set: ',mean_squared_error(Weight,predictions_train,squared = False))

print('R-square of the test set: ',model.score(X_test, y_test))
print('RMSE of the test set: ',mean_squared_error(y_test,predictions_test,squared = False))
