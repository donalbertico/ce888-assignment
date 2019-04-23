from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
import pandas as pd
import numpy as np

#load data
df_1 = pd.read_csv('data/adult.csv')
y = pd.read_csv('data/adult_target.csv',header=None)

#defining classes 0 - > 50k, 1 - < 50k
classes = [0,1]

print(df_1.shape)
print(y.shape)

# splitting dataset
x_train, x_test, y_train, y_test = train_test_split(df_1,y, test_size = 0.2)

# define Random Forest
clf = RandomForestClassifier(n_jobs = 4 ,random_state = 0)
#train random forest
clf.fit(x_train, y_train)

print('Random Forest Report')
print(classification_report(y_pred = clf.predict(x_test),y_true = y_test, labels=classes))

#saving classs probabilities of the whole dataset
probs =clf.predict_proba(df_1)

# encoding probabilities as classes
sofT =  []

for i in range(0,df_1.shape[0]):
    tar = format(probs[i,0]) + "-" + format(probs[i,1])
    sofT.append(tar)

softClass = np.unique(sofT)
print(softClass)

#training a single decision tree on the whole data
simple = tree.DecisionTreeClassifier()
simple.fit(x_train,y_train)

print('Single decision tree performance on whole data')
print(classification_report(y_pred = simple.predict(x_test),y_true = y_test, labels=classes))

#spltting the data with probabilites as new classes
#
dx_train, dx_test, dy_train,dy_test = train_test_split(df_1,sofT, test_size = 0.2)

#training a decistion tree with information distilled
destClf = tree.DecisionTreeClassifier()
destClf.fit(dx_train,dy_train)

#preddicting on test data
preds = destClf.predict(x_test)
translatedPredic = []

#translating predictions from probabilities to classes
for i in range(0,preds.size):
    cla = preds[i].split('-')
    if float(cla[0] > cla[1]):
        translatedPredic.append(0)
    else:
        translatedPredic.append(1)

print('Distilled Tree Forest')
print(classification_report(y_pred = translatedPredic,y_true = y_test, labels=classes))
