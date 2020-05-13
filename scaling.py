import pandas as pd
import numpy as np
from sklearn import svm, preprocessing
import pickle as pickle

fld_select = 'training_noscale.csv'
ddft = pd.read_csv(fld_select)

# shuffle again
ddft=ddft.reindex(np.random.permutation(ddft.index))


X = ddft[['MEAN_TEMPERATURE','MIN_TEMPERATURE','MAX_TEMPERATURE',
         'MAX_REL_HUMIDITY','MIN_REL_HUMIDITY']]
y = ddft['PR_TMMRW']
y=y.astype('int')

# scaling
#############SCALING ON 3 FEATURES###################
scaler = preprocessing.StandardScaler().fit(X)
X=scaler.transform(X)
X=pd.DataFrame(X)
X.columns = ['MEAN_TEMPERATURE','MIN_TEMPERATURE','MAX_TEMPERATURE',
         'MAX_REL_HUMIDITY','MIN_REL_HUMIDITY']
# pickling the scaler object 
filename_p = 'SCALING_FINAL.sav'
pickle.dump(scaler, open(filename_p, 'wb'))

X.to_csv("X.csv")
y.to_csv("y.csv")
 