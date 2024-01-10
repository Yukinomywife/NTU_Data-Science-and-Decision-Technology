# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:16:18 2023

@author: jimmy
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

f = pd.read_excel('Concrete-Homework.xlsx')
df = f.drop(['Unnamed: 0'], axis=1)
df = df.dropna()
df = df.reset_index(drop = True)

df['CoarseAggregate'] = np.log(df['CoarseAggregate'])

train, test = train_test_split(df,test_size = 0.2,random_state = 4172888)

df1 = pd.DataFrame(train, columns = df.columns)
df2 = pd.DataFrame(test, columns = df.columns)

y_train = df1.iloc[:,-1]
X_train = df1.iloc[:,:len(df1.columns)-1]
y_test = df2.iloc[:,-1]
X_test = df2.iloc[:,:len(df2.columns)-1]

model = Sequential()
model.add(Dense(units=36768, activation = 'relu', input_dim=(X_train.shape[1]),\
                kernel_regularizer=l2(0.2),\
                bias_regularizer =l1(0.2),\
                activity_regularizer = l1_l2(l1=0.002, l2=0.002)))
model.add(Dense(units=1))

optimizer = Adam(learning_rate=4e-5)

model.compile(loss='mean_squared_error', optimizer=optimizer)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=5000,verbose=0,\
          validation_split=0.1, batch_size=256,callbacks = [early_stopping])

y_pred = model.predict(X_test,batch_size=2,verbose=1)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)