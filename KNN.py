# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 00:22:19 2023

@author: Mustafa
"""

import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')

diabets = pd.read_csv("C:/Users/Mustafa/makine_ogrenmesi/deneme/diabetes.csv")
df=diabets.copy()
df= df.dropna()
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)
print(df.head())
df.info()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.25,random_state=42)

knn=KNeighborsClassifier()
knn_model = knn.fit(X_train,y_train)
y_pred=knn_model.predict(X_test)
print()

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred) ## neyle ne arasında confusion matrix olusturucan soruyor bizde tahminler üzerinden olusturacagız 
print(cm)

accuracy_score(y_test, y_pred) ## 1  e ne kadar yakındsa o kadar iyi sınıflandırma demek 
print(accuracy_score(y_test, y_pred)) ##  bu bizim ilkel skorumuz dogru sınıflandırma oranı 1 e yakın oldukca sınıflandır basarısı o kadar iyi demek 

print(classification_report(y_test, y_pred)) ## detaylı cıktı 

## Model tunning yaptık burada dogrulugu kaç komsuluk yaparsak daha iyi olur ölcüp denedik ve artırdık verimi
knn_params={"n_neighbors" : np.arange(1,50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn, knn_params,cv=10)
knn_cv.fit(X_train,y_train)
print("En iyi skor : " + str(knn_cv.best_score_))
print("En iyi parametre : " +str(knn_cv.best_params_))

knn=KNeighborsClassifier(n_neighbors=8)
knn_tuned=knn.fit(X_train, y_train)
y_pred9=knn_tuned.predict(X_test)

print(accuracy_score(y_test, y_pred9))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred9) ## neyle ne arasında confusion matrix olusturucan soruyor bizde tahminler üzerinden olusturacagız 
print(cm)


