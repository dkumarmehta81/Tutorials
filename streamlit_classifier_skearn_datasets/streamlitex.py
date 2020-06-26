# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:39:31 2020

@author: princ
"""

import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

def getClassifierParams(clfname):
    param=dict()
    if clfname=='KNN':
        K=st.sidebar.slider("K",1,10)
        param["K"]=K 
        
    elif clfname=='RandomForest':
        nestimators=st.sidebar.slider("nestimator",10,100)
        maxdepth=st.sidebar.slider("maxdepth",1,10)
        param["nestimators"]=nestimators
        param["maxdepth"]=maxdepth
        
    else:
        C=st.sidebar.slider("C",0.1,10.0)
        param["C"]=C
    return param

def getDataset(dsn):
    if dsn=='Wine':
        data=datasets.load_wine()
    elif dsn=='Iris':
        data=datasets.load_iris()
    else:
        data=datasets.load_breast_cancer()
    X=data.data
    y=data.target
    return X,y    



def getClassifier(clfname,param):
    clf=None
    if clfname=='KNN':
        clf=KNeighborsClassifier(n_neighbors=param["K"])
    elif clfname=='RandomForest':
        clf=RandomForestClassifier(n_estimators=param["nestimators"],max_depth=param["maxdepth"])
    else:
        clf=SVC(C=param["C"])
    return clf

st.title("streamlit example")
st.write("""
         #  explore different classifier
         which one is best
         
         
         """)
dsn=st.sidebar.selectbox("Select Datset",("Iris","Wine","BreastCancer"))
st.write(dsn)
X,y=getDataset(dsn)

st.write("Shape",X.shape)
st.write("number of classes",len(np.unique(y)))


classifiername=st.sidebar.selectbox("Select Classifier",("KNN","SVM","RandomForest"))
param=getClassifierParams(classifiername)
clf=getClassifier(classifiername,param)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=0)
clf.fit(xtrain,ytrain)
ypred=clf.predict(xtest)
acc=accuracy_score(ytest,ypred)
cm=confusion_matrix(ytest,ypred)
st.write("Classifier Name={}".format(classifiername))
st.write("Classification Accuracy={}".format(acc))
#st.write("Confusion Matrix={}".format(cm))  

pca=PCA(2)
x_projected=pca.fit_transform(X)
x1=x_projected[:,0]
x2=x_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis') 
plt.xlabel("PCA1")
plt.ylabel("PCA2")    
plt.title(dsn)
plt.colorbar()
st.pyplot()
