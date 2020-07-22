# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 20:31:00 2020

@author: Pierce
"""
# lineman binary classification
# Feature selection
# Model Additional Ensemble Approaches
# confusion matrix 

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r'C:/Users/Pierce/Desktop/DS/Football Pos Scraper/football_preprocessed_df.csv')

# convert position to lineman (yes or no)
def lineman_converter(x):
    if x == 'OL' or x == 'DL':
        return 1
    else: 
        return 0
df["lineman_yn"] = df["position"].apply(lineman_converter)

df_model = df[['lineman_yn', 'weight',
       'forty_yard_dash', 'shuttle_run', 'three_cone', 'broad_jump',
       'vertical_jump', 'grade', 'height_inches']]
df_model['shuttle_run'] = df['shuttle_run'].apply(pd.to_numeric, errors = 'coerce')

df_model = df_model.dropna()

from sklearn.model_selection import train_test_split
X = df_model.iloc[:, 1:].values
y = df_model.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardization
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train_std = scale.fit_transform(X_train)
X_test_std = scale.fit_transform(X_test)

# naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
cv = cross_val_score(gnb,X_train_std,y_train,cv=5)
print(cv)
print(cv.mean())
# 88.4%

# logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
cv = cross_val_score(lr,X_train_std,y_train,cv=5)
print(cv)
print(cv.mean())
# 90.7%

# Decision Tree
from sklearn import tree
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train_std,y_train,cv=5)
print(cv)
print(cv.mean())
# 85.4%

# K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
cv = cross_val_score(knn,X_train_std,y_train,cv=5)
print(cv)
print(cv.mean())
# 89.5%

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train_std,y_train,cv=5)
print(cv)
print(cv.mean())
# 90.5%

# Support Vector Classifier
from sklearn.svm import SVC
svc = SVC(probability=True)
cv = cross_val_score(svc,X_train_std,y_train,cv=5)
print(cv)
print(cv.mean())
# 90.7%

# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1)
cv = cross_val_score(xgb,X_train_std,y_train,cv=5)
print(cv)
print(cv.mean())
# 89.5%

# soft voting classifier
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gnb',gnb),('svc',svc),('xgb',xgb)], voting = 'soft') 
cv = cross_val_score(voting_clf,X_train_std,y_train,cv=5)
print(cv)
print(cv.mean())
# 90.5%

# Tuning models
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 

#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))

# tuning logistic regression 
lr = LogisticRegression()
param_grid = {'max_iter' : [100,2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear', 'newton-cg', 'lbfgs']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train_std,y_train)
clf_performance(best_clf_lr,'Logistic Regression')
# 90.7%

# tuning KNN
knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : [3,5,7,9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]}
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train_std,y_train)
# 90.2%

# tuning random forest
rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [100,500,1000], 
                                  'bootstrap': [True,False],
                                  'max_depth': [3,5,10,20,50,75,100,None],
                                  'max_features': ['auto','sqrt'],
                                  'min_samples_leaf': [1,2,4,10],
                                  'min_samples_split': [2,5,10]}
                                  
clf_rf_rnd = RandomizedSearchCV(rf, param_distributions = param_grid, n_iter = 100, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf_rnd = clf_rf_rnd.fit(X_train_std,y_train)
print('Best Score: ' + str(best_clf_rf_rnd.best_score_))
print('Best Parameters: ' + str(best_clf_rf_rnd.best_params_))
# 90.7%

rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [500,550],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [75,100],
                                  'max_features': ['auto','sqrt'],
                                  'min_samples_leaf': [2,4],
                                  'min_samples_split': [2,5,10]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train_std,y_train)
clf_performance(best_clf_rf,'Random Forest')
# 90.7%

# Model Additional Ensemble Approach
best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_rf = best_clf_rf.best_estimator_
from sklearn.ensemble import VotingClassifier
voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf), ('lr', best_lr)],voting = 'soft') 
print('voting_clf_soft :',cross_val_score(voting_clf_soft,X_train_std,y_train,cv=5))
print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X_train_std,y_train,cv=5).mean())
# 90.7%


# test data - baseline logistic regression
lr.fit(X_train_std, y_train)
lr.predict(X_test_std)
lr.score(X_test_std, y_test)
# 91.9%