import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV 
from sklearn.decomposition import PCA 
from sklearn.metrics import confusion_matrix

fld_select = 'training_noscale.csv'
fld_new = 'hum_train_final.csv'

# pre-processing
dft = pd.read_csv(fld_new)
count=dft.count().sort_values()

dft = dft.drop(columns=['STATION_NAME','STN_ID','LOCAL_DATE', 'CLIMATE_IDENTIFIER','ID','PROVINCE_CODE',
                     'LOCAL_YEAR','LOCAL_MONTH','LOCAL_DAY','MEAN_TEMPERATURE_FLAG','MIN_TEMPERATURE_FLAG',
                      'MAX_TEMPERATURE_FLAG','TOTAL_PRECIPITATION_FLAG','TOTAL_RAIN','TOTAL_RAIN_FLAG',
                      'TOTAL_SNOW','TOTAL_SNOW_FLAG','SNOW_ON_GROUND','SNOW_ON_GROUND_FLAG',
                      'DIRECTION_MAX_GUST_FLAG','SPEED_MAX_GUST_FLAG','COOLING_DEGREE_DAYS_FLAG',
                      'HEATING_DEGREE_DAYS_FLAG','MIN_REL_HUMIDITY_FLAG','MAX_REL_HUMIDITY_FLAG','x','y',
                      'Unnamed: 0','Unnamed: 0.1'],axis=1)
print(dft.columns) 
sns.heatmap(dft.isnull(),yticklabels=False,cbar=False,cmap='Reds_r')
  
#-9999
dft=dft.replace(np.nan,-9999)
dft=dft.reindex(np.random.permutation(dft.index))
 
########REMOVE OUTLIERS####################
z = np.abs(stats.zscore(dft._get_numeric_data()))
dft= dft[(z < 3).all(axis=1)]

#dft.to_csv('training_noscale.csv')

# feature selection
ddft = pd.read_csv(fld_select)

# shuffle again
ddft=ddft.reindex(np.random.permutation(ddft.index))


X = ddft[['MEAN_TEMPERATURE','MIN_TEMPERATURE','MAX_TEMPERATURE','TOTAL_PRECIPITATION',
         'MAX_REL_HUMIDITY','MIN_REL_HUMIDITY','TOTAL_PRECIPITATION','DIRECTION_MAX_GUST',
         'SPEED_MAX_GUST','COOLING_DEGREE_DAYS', 'HEATING_DEGREE_DAYS','PR_TODAY']]

y = ddft['PR_TMMRW']
y=y.astype('int')


#########1. FEATURE SELECTION- RFE##############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=88)

clf0 = DecisionTreeClassifier()
rfe = RFE(estimator=clf0, n_features_to_select=13)
rfe = rfe.fit(X_train,y_train)
print("Number of Features: {}".format(rfe.n_features_)) 
print("Selected Features: {}".format(rfe.support_))
print("Feature Ranking: {}".format(rfe.ranking_))
pd.DataFrame(X.iloc[:,rfe.support_].columns,columns=['Importance'])


####2. FEATURE SELECTION PCA##########
# Initializing PCA and fitting
pca = PCA()
pca.fit(X_train)

# Plotting to visualize the best number of elements
plt.figure(1, figsize=(9, 8))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('Number of Feautres')
plt.ylabel('Variance Ratio')

#######3. RFE WITH CROSS-VALIDATION#############
# Initialize the decision tree Classifier
DecisionTree_RFECV = DecisionTreeClassifier() 
# Initialize the RFECV function setting 3-fold cross validation
rfecv = RFECV(estimator=DecisionTree_RFECV, step=1, cv=3, scoring='accuracy')
# Fit data
rfecv = rfecv.fit(X_train, y_train)

print('Best number of features :', rfecv.n_features_)
print('Features :\n')
for i in X_train.columns[rfecv.support_]:
    print(i)

# Plotting the best features with respect to the Cross Validation Score
plt.figure()
plt.xlabel("Number of Features")
plt.ylabel("Score of Selected Features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


########4.RFE AGAIN########################
# Initializing decision tree Classifier
DecisionTree_RFE = DecisionTreeClassifier() 
# Initializing the RFE object, one of the most important arguments is the estimator, in this case is RandomForest
rfe = RFE(estimator=DecisionTree_RFE, n_features_to_select=5, step=1)
# Fit the origial dataset
rfe = rfe.fit(X_train, y_train)

print("Best features chosen by RFE: \n")
for i in X_train.columns[rfe.support_]:
    print(i)
    
# Generating X_train and X_test based on the best features given by RFE
X_train_RFE = rfe.transform(X_train)
X_test_RFE = rfe.transform(X_test)
# Fitting the decision tree
DecisionTree_RFE = DecisionTree_RFE.fit(X_train_RFE, y_train)

#Making a prediction and calculting the accuracy
y_pred = DecisionTree_RFE.predict(X_test_RFE)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ',accuracy)

# Showing performance with a confusion matrix
confMatrix = confusion_matrix(y_test, y_pred)
sns.heatmap(confMatrix, annot=True, fmt="d")





