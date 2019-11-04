import pandas as pd # for data analytics
import numpy as np # for numerical computation
from sklearn.metrics import precision_recall_fscore_support, classification_report,confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn import utils  
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('breast_cancer_classification.csv')
df.loc[df['diagnosis'] == 'M' , "Class"] = -1
df.loc[df['diagnosis'] == 'B' , "Class"] = 1
#Delea a coluna com Id
del(df['id'])
del(df['diagnosis'])


#getting random set of nonfraud data to train on
non_fraud = df[df['Class']==1]
df_train, val = train_test_split(non_fraud, test_size=0.20, random_state=42)
fraud = df[df['Class']==-1]

#Aplica o escalonamento
scaler = StandardScaler()
df = scaler.fit_transform(df.iloc[:,:])

model = OneClassSVM(kernel='rbf', nu=0.0005,gamma=0.007)
model.fit(df)

#Creating a test set that contains both fraud and non fraud
y_val = val['Class']
y_fraud = fraud['Class']
y_testval = pd.concat([y_val, y_fraud])
y_testval = np.array(y_testval)
df_testval = pd.concat([val, fraud])

#predicting on test set, which consists of both fraud and non-fraud
pred_testval = model.predict(df_testval)

print(classification_report(y_testval, pred_testval))

prec, rec, f2, _ = precision_recall_fscore_support(y_testval, pred_testval, beta=2, 
                                                   pos_label=-1, average='binary')
print(f'precision is {prec}, recall is {rec} and F2 score is {f2}')

roc = roc_auc_score(y_testval, pred_testval)
print(f'ROC score is {roc}')
