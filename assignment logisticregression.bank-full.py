# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 21:32:34 2022

@author: vaishnav
"""

import pandas as pd

df = pd.read_csv("C:\\anaconda\\bank-full.csv",sep = ';')
list(df)

df.isnull().sum()

df.dtypes
df.shape


#=====================================================================================
import seaborn as sns

df["age"].hist()
df["age"].skew()

df["balance"].hist()
df["balance"].skew()

df["day"].hist()
df["day"].skew()

df["duration"].hist()
df["duration"].skew()

df["campaign"].hist()
df["campaign"].skew()

df["pdays"].hist()
df["pdays"].skew()

df["previous"].hist()
df["previous"].skew()
#=================================================================================

# Visualise count plot
sns.countplot(data=df,x='y')


import matplotlib.pyplot as plt
plt.scatter(x=df["age"],y=df["balance"])

plt.scatter(x=df["day"],y=df["duration"])

plt.scatter(x=df["campaign"],y=df["pdays"])

plt.scatter(x=df["previous"],y=df["balance"])

#================================================================================
#spliting into contineous and categorical

df_cont = df[df.columns[[0,5,9,11,12,13,14]]]


df_cat = df[df.columns[[1,2,3,4,6,7,8,10,15,16]]]

#=================================================================================
#data transformation
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df_cat:
    # create an object of LabelEncoder
    le=LabelEncoder()
    df_cat[col]=le.fit_transform(df_cat[col])
df_cat.dtypes

#====================================================================================

df_new = pd.concat([df_cont,df_cat],axis=1)

df_new.head()

#=======================================================================================
# Split dataset in input and output
X=df_new.drop('y',axis=1)    # input
Y=df_new['y']                # output
X.head()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_x = ss.fit_transform(X)


#==========================================================================================
#split the data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(ss_x,Y,test_size=0.33,random_state=(42))

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression()

LogReg.fit(X_train,Y_train)
y_pred_train = LogReg.predict(X_train)
y_pred_test = LogReg.predict(X_test)


# Generation Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred_test))

from sklearn.metrics import accuracy_score
print("accuracy score for test:",accuracy_score(Y_test,y_pred_test).round(2))
#============================================================================================
