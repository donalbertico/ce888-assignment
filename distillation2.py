from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree


import pandas as pd
import numpy as np

df_1 = pd.read_csv('data/art_char.csv')
y = pd.read_csv('data/art_char_target.csv',header=None)

classes = np.unique(y)

print(df_1.head)
print(classes)


x_train, x_test, y_train, y_test = train_test_split(df_1,y, test_size = 0.3)

clf = RandomForestClassifier(n_jobs = 4 ,random_state = 0)

clf.fit(x_train, y_train)

# print(classification_report(y_pred = clf.predict(x_test),y_true = y_test, labels=classes))

probs =clf.predict_proba(df_1)

destiled = df_1.copy()
destiled['>'] = probs[:,0]
destiled['<'] = probs[:,0]

sofT =  []

for i in range(0,df_1.shape[0]):
    tar = format(probs[i,0]) + "-" + format(probs[i,1])
    sofT.append(tar)

softClass = np.unique(sofT)
# print(destiled)

simple = tree.DecisionTreeClassifier()
simple.fit(x_train,y_train)

print(classification_report(y_pred = simple.predict(x_test),y_true = y_test, labels=classes))



dx_train, dx_test, dy_train,dy_test = train_test_split(df_1,sofT, test_size = 0.3)



destClf = tree.DecisionTreeClassifier()
destClf.fit(dx_train,dy_train)

preds = destClf.predict(dx_test)
predic = []

for i in range(0,preds.size):
    cla = preds[i].split('-')

    if float(cla[0] > cla[1]):
        predic.append(0)
    else:
        predic.append(1)

print(classification_report(y_pred = predic,y_true = y_test, labels=classes))
