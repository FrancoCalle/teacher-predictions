import pandas as pd
import numpy as np
import os
%matplotlib inline
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.plotly as py

df_path = os.getcwd() + '\data\entrancescore-evdocente-noid.csv'
df = None
#with open(df_path, encoding="utf-16") as f:
with open(df_path, "r") as f:
    df = pd.read_csv(f)

list(df)
df.head(3)


#Divide the sample distribution in 10 types:
df['xtile'] = pd.qcut(df.pf, 10, labels = ['percentil: '+str(i) for i in range(10)])
df['worst'] = (df['xtile'] == 'percentil: 0' )| (df['xtile'] == 'percentil: 1')| (df['xtile'] == 'percentil: 2')


#Create a histogram:
list(df)

#Generate graphs: Histograms
pyplot.hist((df.paaverbal.dropna()-df.paaverbal.mean())/df.paaverbal.std(), 100, alpha=0.5, label='PAA - Verbal')
pyplot.hist((df.pf.dropna()-df.pf.mean())/df.pf.std(), 100, alpha=0.5, label='PJ - Portfolio')
pyplot.legend(loc='upper right')
pyplot.show()

#Generate graphs: Scatters:
#NEM vs PF_PJE:
df1 = df.sort_values(by = ['pf_pje'])
df1['xtile'] = pd.qcut(df1.pf_pje, 50, labels = list(range(50)))
scatter1 = df1[['pf', 'paaverbal', 'paamat']].groupby(df1['xtile']).mean()
plt.scatter((scatter1.paaverbal.dropna()-scatter1.paaverbal.mean())/scatter1.paaverbal.std(), (scatter1.pf.dropna()-scatter1.pf.mean())/scatter1.pf.std(), c="r", alpha=0.5, label="Correlation")
plt.xlabel("PAA - Verbal")
plt.ylabel("PJ - Portfolio")
plt.legend(loc=2)
plt.show()

df['paaverbal'].isnull().sum()
df['paamat'].isnull().sum()



#Generate Machine Learning Models:
#Using regression trees:
from sklearn import datasets, linear_model, preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree, svm
from sklearn.metrics import accuracy_score, roc_auc_score

df = df.dropna()
X = df[['paaverbal','paamat']].dropna()
Y = df['worst']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train_transformed = preprocessing.StandardScaler().fit(X_train)
X_test_transformed = preprocessing.StandardScaler().fit(X_test)
X_train_transformed = X_train_transformed.transform(X_train)
X_test_transformed  = X_test_transformed.transform(X_test)


#1. Decision Tree Classifier:
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train_transformed, y_train)

clf.score(X_train_transformed, y_train)
clf.score(X_test_transformed, y_test)

clf = make_pipeline(preprocessing.StandardScaler(), tree.DecisionTreeClassifier())
accuracy_rt = cross_val_score(clf, X, Y, cv=500)

pyplot.hist(accuracy_rt, 30, alpha=0.5, label='Density', color = 'g')
pyplot.legend(loc='upper right')
pyplot.ylabel('Density')
pyplot.xlabel('Accuracy Score')
pyplot.show()


#1. Support Vector Machine

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
accuracy_svm = cross_val_score(clf, X, Y, cv=50, n_jobs = 4)

pyplot.hist(accuracy_svm, 7, alpha=0.5, label='Density')
pyplot.legend(loc='upper right')
pyplot.ylabel('Density')
pyplot.xlabel('Probability')
pyplot.show()


clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=10))
accuracy_svm = cross_val_score(clf, X, Y, cv=2)
pyplot.hist(accuracy_svm, 7, alpha=0.5, label='Density')
pyplot.legend(loc='upper right')
pyplot.ylabel('Density')
pyplot.xlabel('Probability')
pyplot.show()



#Random Forests
from sklearn.ensemble import RandomForestClassifier
clf = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier(n_jobs=10, random_state=0))
accuracy_rf = cross_val_score(clf, X, Y, cv=100)
pyplot.hist(accuracy_rf, 7, alpha=0.5, label='Density')
pyplot.legend(loc='upper right')
pyplot.ylabel('Density')
pyplot.xlabel('Probability')
pyplot.show()


#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = make_pipeline(preprocessing.StandardScaler(), LogisticRegression())
accuracy_log = cross_val_score(clf, X, Y, cv=1000)
pyplot.hist(accuracy_log, 7, alpha=0.5, label='Density')
pyplot.legend(loc='upper right')
pyplot.ylabel('Density')
pyplot.xlabel('Probability')
pyplot.show()



#Lasso Regression
from sklearn.linear_model import LogisticRegression
clf = make_pipeline(preprocessing.StandardScaler(), LogisticRegression())
accuracy_log = cross_val_score(clf, X, Y, cv=100)
pyplot.hist(accuracy_log, 7, alpha=0.5, label='Density')
pyplot.legend(loc='upper right')
pyplot.ylabel('Density')
pyplot.xlabel('Probability')
pyplot.show()



clf.predict_proba()
