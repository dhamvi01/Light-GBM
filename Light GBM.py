import pandas as pd
import numpy as np
pd.set_option('display.max_columns',200)
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

#---------------------------------------------------------------------------------------------|

data = pd.read_csv("C:\\Users\\vijaykumar.dhameliya\\Desktop\\1000P\\5. Light GBM\\train.csv",skiprows=1,header=None)
data.columns = ['age','workclass','fnlwgt','education','education-num','marital_Status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','Income'] 

print(data.head())


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l = LabelEncoder()
data.Income = l.fit_transform(data.Income)

v = pd.get_dummies(data[['workclass','education','marital_Status','occupation','relationship','race','sex','native_country']])
w = data.drop(['workclass','education','marital_Status','occupation','relationship','race','sex','native_country'],axis=1)

new_data = v.join(w)

_,i = np.unique(new_data.columns, return_index=True)
new_data = new_data.iloc[:,i]

x = new_data.drop('Income', axis=1)
y = new_data.Income

y.fillna(y.mode()[0], inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#-------------------------------------------------------------------------------------------------
###-------------------------------------- X gboost ---------------------------------------------->
#-------------------------------------------------------------------------------------------------

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest  = xgb.DMatrix(x_test)

parameter = {'max_depth' : 7, 'eta' : 1, 'silent':1, 'objective' : 'binary:logistic', 'eval_matric' : 'auc', 'learning_rate' : 0.05}

num_round = 50
from datetime import datetime
start = datetime.now()
xg = xgb.train(parameter,dtrain,num_round)
stop = datetime.now()

execution_time_xgb = stop - start
print('--'*20,execution_time_xgb,'--'*20)

ypred = xg.predict(dtest)
print(ypred)

print(ypred.shape[0])

for i in range(ypred.shape[0]):
    if ypred[i] > 0.5:
        ypred[i] = 1
    else:
        ypred[i] = 0

from sklearn.metrics import accuracy_score, confusion_matrix

acc_xgb = accuracy_score(y_test,ypred)
print(confusion_matrix(y_test, ypred))

#------------------------------------------------------------------------------------------------------
#------------------------------------ LIGHT GBM ------------------------------------------------------>
#------------------------------------------------------------------------------------------------------

train_dataset = lgb.Dataset(x_train,label=y_train)

param = {'num_leaves' : 150, 'objective' : 'binary', 'max_depth' : 7, 'learning_rate' : 0.05, 'max_bin' : 200}
param['metric'] = ['auc', 'binary_logloss']

num_round = 50
from datetime import datetime
start = datetime.now()
lgb = lgb.train(param,train_dataset,num_round)
stop = datetime.now()

execution_time_lgb = stop - start
print('--'*20,execution_time_lgb,'--'*20)

ypred2 = lgb.predict(x_test)
print(ypred2)

for i in range(ypred2.shape[0]):
    if ypred2[i] > 0.5:
        ypred2[i] = 1
    else:
        ypred2[i] = 0

lgb_xgb = accuracy_score(y_test,ypred2)
print(confusion_matrix(y_test, ypred2))

#||----------------------------------------------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score

xgb_auc = roc_auc_score(y_test,ypred)
lgb_auc = roc_auc_score(y_test,ypred2)

comparision_dict = {'accuracy score' : [acc_xgb,lgb_xgb],'auc score' : [xgb_auc,lgb_auc,],'execution time' : [execution_time_xgb,execution_time_lgb]}

comparision_df = pd.DataFrame(comparision_dict)
comparision_df.index = ['Xgboost', 'LighGBM']
print(comparision_df)







