# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 22:40:09 2019

@author: Ankuran Das
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#loading the dataset
data=pd.read_csv('CC.csv')

#exploring the dataset
#print(data.columns)
#print(data.shape)
#print(data.describe())

data=data.sample(frac=0.1,random_state=1)
print(data.shape)

#plot a histogram
plt.show()

#determine the number of fraud cases in dataset
fraud=data[data['Class']==1]
valid=data[data['Class']==0]

outliner_fraction= len(fraud) / float(len(valid))

#corelation matrix
corrmat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax= 0.8,square=True)
plt.show()

#get all the columns from the dataframe
columns=data.columns.tolist()

#fliter the columns to remove data we donot want
columns=[c for c in columns if c not in["Class"]]

#store the variable we'll be predicting on
target="Class"

X=data[columns]
Y=data[target]

#define a random state
state=1

#define the outlier detection methods
classifiers={
        "Isolation Forest":IsolationForest(max_samples=len(X),
                                           contamination=outliner_fraction,
                                           random_state=state
        ),
        "Local Outlier Fraction":LocalOutlierFactor(
                                        n_neighbors=20,
                                        contamination=outliner_fraction
                )
        
        }
#fit the model
n_outliers=len(fraud)
for i,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name=="local Outlier Factor":
        y_pred=clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
        
    #Reshape the prediction 0 for valid 1 for fraud
    
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    
    n_error=(y_pred!=Y).sum()
    
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
    
