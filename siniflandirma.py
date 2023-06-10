# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x=veriler.iloc[:,1:4].values  ##bagımsız degiskenler
y=veriler.iloc[:,4:].values ## bagımlı değişken
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train) ## fit eğitme transformda ise o eğitimi kullanma uygula yani
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)

logr.fit(X_train,y_train) # x ten y yi ögrenicez eğittik y_train kadın erkek olmasını öğrenmesini soyledik 

y_pred =logr.predict(X_test) 
print(y_pred)
print(y_test)

## Confusion matrix kullandık 
print("LogisticReg Confusion Matrix")
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred) ## neyle ne arasında confusion matrix olusturucan soruyor bizde tahminler üzerinden olusturacagız 
print(cm)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')  ## kac komsuya bakıcak metric = mesagfe 
knn.fit(X_train,y_train)

y_pred1=knn.predict(X_test)
print("Knn Confusion Matrix")
cm=confusion_matrix(y_test, y_pred1)
print(cm)

## Model tunning baktın yani komsulukta en iyi skor neyi verir ona baktık
knn_params={"n_neighbors" : np.arange(1,50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn, knn_params,cv=10)
knn_cv.fit(X_train,y_train)
print("En iyi skor : " + str(knn_cv.best_score_))
print("En iyi parametre : " +str(knn_cv.best_params_))

knn=KNeighborsClassifier(n_neighbors=1)
knn_tuned=knn.fit(X_train, y_train)
y_pred9=knn_tuned.predict(X_test)



## support vector regression
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred2=svc.predict(X_test)

cm=confusion_matrix(y_test, y_pred2)
print('SVC')
print(cm)

##Model tunning yaptık baktık kaç C sine vs baktık 
svm_params={"C": np.arange(1,10)}
svm=SVC(kernel="rbf")
svm_cv=GridSearchCV(svm, svm_params,cv=10)
svm_cv.fit(X_train,y_train)
print("En iyi skor : " + str(svm_cv.best_score_))
print("En iyi parametre : " +str(svm_cv.best_params_))

knn=KNeighborsClassifier(n_neighbors=1)
knn_tuned=knn.fit(X_train, y_train)
y_pred9=knn_tuned.predict(X_test)


## Naive  Bases 

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train, y_train)
y_pred3=gnb.predict(X_test)

cm=confusion_matrix(y_test, y_pred3)
print('GNB')
print(cm)


## Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)
y_pred4 = dtc.predict(X_test)

cm=confusion_matrix(y_test, y_pred4)
print('DTC')
print(cm)


##Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train, y_train)
y_pred5 = rfc.predict(X_test)
cm=confusion_matrix(y_test, y_pred5)
print('RFC')
y_proba=rfc.predict_log_proba(X_test)
print(cm)
print(y_proba)




