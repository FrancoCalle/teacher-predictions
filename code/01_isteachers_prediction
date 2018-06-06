import pandas as pd
import numpy as np
import os
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.plotly as py
from scipy import stats
from sklearn import datasets, linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree, svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LinearRegression

#df_path = os.getcwd() + '\data\entrancescore-evdocente-noid.csv'
df_path = os.getcwd() + '\data\PAA-evdoc.csv'
df = None
#with open(df_path, encoding="utf-16") as f:
with open(df_path, "r") as f:
    df = pd.read_csv(f)


q_v    = df["paa_verbal"].quantile(0.95)
q_m = df["paa_matematica"].quantile(0.95)

df = df[(df.paa_verbal<q_v) | (df.paa_matematica<q_m)]

#Quick df clean:
scores = ['paa_verbal', 'paa_matematica', 'nem', 'gpa', 'pce_hria_y_geografia', 'pce_biologia', 'pce_cs_sociales', 'pce_fisica', 'pce_matematica', 'pce_quimica']
cat = ['region', 'male']
df_s = df[scores].copy()
x_mean = df[scores].mean()
x_std = df[scores].std()
df_s = (df[scores] - df[scores].mean())/df[scores].std()
df_s['Average'] = (df['paa_verbal'] + df['paa_matematica'])/2

df_c = df[cat].copy()

df_s['took_hist'] = (df.pce_hria_y_geografia == 0)
df_s['took_bio'] = (df.pce_biologia == 0)
df_s['took_soc'] = (df.pce_cs_sociales == 0)
df_s['took_fis'] = (df.pce_fisica == 0)
df_s['took_mat'] = (df.pce_matematica == 0)
df_s['took_qui'] = (df.pce_quimica == 0)

y = df['pf_pje']

df = pd.concat([df_s, df_c, y], axis=1, sort=False)

df = df.dropna()
df.region = df.region.astype(int)
df.male = df.male.astype(int)

#Generate dichotomous variables by region
for x in set(df.region):
    df['region' + str(x)] = df.region == x

#Divide the sample distribution in 10 types:
df['xtile'] = pd.qcut(df.pf_pje, 10, labels = ['percentil: '+str(i) for i in range(10)])
df['worst'] = (df['xtile'] == 'percentil: 0' ) | (df['xtile'] == 'percentil: 1') | (df['xtile'] == 'percentil: 2')

df.columns = ['paaverbal' if x=='paa_verbal' else 'paamat' if x =='paa_matematica' else x for x in df.columns] #Change colname

x_variables = [x if (x!='pf_pje' and x!='xtile' and x!='worst' and x!='region9') else "AAA" for x in list(df)]
x_variables = sorted(list(set(x_variables)))
del x_variables[0]

X_transformed = df[x_variables].dropna().drop(columns = ['region',
'pce_biologia', 'pce_cs_sociales', 'pce_fisica', 'pce_hria_y_geografia', 'pce_matematica', 'pce_quimica',
'region1', 'region2', 'region3', 'region4', 'region5', 'region6', 'region7', 'region8', 'region10', 'region11', 'region12', 'region13', 'region14', 'region15'])
#'took_bio', 'took_fis', 'took_hist', 'took_mat', 'took_qui', 'took_soc', 'paamat', 'paaverbal'

X_transformed['paa_avg'] = (X_transformed.Average - X_transformed.Average.mean())/X_transformed.Average.std()

Y = df['worst']

X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed.drop(columns = ['Average', 'paa_avg', 'gpa']), Y, test_size=0.15)

resampling = 0
if resampling == 1:
    Train_set = X_train_transformed.copy()
    Test_set = X_test_transformed.copy()
    Train_set['y_train'] = y_train
    Train_set['random'] = np.random.randn(X_train_transformed.shape[0],1)
    n_true = Train_set[Train_set.y_train == 1].shape[0]
    n_false = Train_set[Train_set.y_train == 0].shape[0]
    n_start = n_false - n_true
    Train_set = Train_set.sort_values(by = ['y_train','random']).reset_index().iloc[n_start:]

    x_variables = [x if (x!='index' and x!='random' and x!='y_train' and x!='region9' ) else 'AAA' for x in list(Train_set)]
    x_variables = sorted(list(set(x_variables)))
    del x_variables[0]
    X_train_transformed = Train_set[x_variables]
    X_test_transformed = Test_set[x_variables]
    y_train = Train_set.y_train


import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\Franco\\Anaconda3\\Library\\bin\\graphviz'
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_leaf_nodes = 8)
dtree.fit(X_train_transformed,y_train)
#dtree.fit(X_train_transformed.drop(columns = ['paamat', 'paaverbal', 'male']),y_train)

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())



#Define some useful functions:
#============================

def gen_predictions(Classifier, X_train, X_test, y_train):
    from sklearn import tree, svm
    SVC = svm.SVC
    if Classifier == SVC:
        clf = Classifier(C=1, probability=True)
    else:
        clf = Classifier()
    clf = clf.fit(X_train, y_train)

    y_test_hat = clf.predict_proba(X_test)[:,1]  #Prob of bad Teacher
    y_test_predict = clf.predict(X_test)  #Prob of bad Teacher

    ndf = y_test_hat.shape[0]
    df_ml_test = X_test.copy()
    df_ml_test['y_test_hat'] = y_test_hat
    df_ml_test = df_ml_test.sort_values(by = ['y_test_hat'])

    return df_ml_test, clf.predict_proba(X_test), y_test_predict


# Options: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier svm.SVC
df_rf_test , y_test_hat, y_test_predict = gen_predictions(RandomForestClassifier, X_train_transformed, X_test_transformed, y_train)
#df_rf_test.to_csv(r'C:\Users\Franco\GitHub\teacher-predictions\output\data_predictions.csv')
