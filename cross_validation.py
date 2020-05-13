import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os, sys
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

input="/Users/shawndegroot/Documents/Shawn/ML/Raintmmrw/Rain Cad/scripts/humidity/final13/"

fld1='X.csv'
fld2='y.csv'
X=pd.read_csv(fld1)
X = X.drop(["Unnamed: 0"], axis=1)
y=pd.read_csv(fld2)
y = y.drop("Unnamed: 0", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=88)
  # prepare configuration for cross validation test harness
seed = 7
#  # prepare models
models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('SGD', SGDClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))

#evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

#boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
fig.set_figheight(7)
fig.set_figwidth(14)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



