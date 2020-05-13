import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True) 

fld1='X.csv'
fld2='y.csv'
X=pd.read_csv(fld1)
X = X.drop(["Unnamed: 0"], axis=1)

y=pd.read_csv(fld2)
y = y.drop("Unnamed: 0", axis=1)

##########TRAINING/TESTING/MODEL ACCURACY##############
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=88)
clf= DecisionTreeClassifier()

t0=time.time()

# actual train
clf.fit(X_train, y_train)

###-SAVING TRAINED MODEL TO DISK##
filename = 'DECISION_TREE_FINAL.sav'
pickle.dump(clf, open(input+filename, 'wb'))

y_pred=clf.predict(X_test)

# actual test
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:" +str(accuracy * 100)+" %")
print('Time taken :' , time.time()-t0)

