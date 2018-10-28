import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
import sys
from time import time
data = pd.read_csv("./train.csv")

used_features =[
	"AP1",
    "AP2",
    "AP3",
    "AP4","AP5","AP6","AP7"
	]
features = data[used_features]
labels = data["ROOM"]

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print "Gaussian naive_bayes :"
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
print clf.score(features_test,labels_test)



print "KNN :"
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(n_neighbors=9,algorithm='ball_tree')
clf1.fit(features_train,labels_train)
print clf1.score(features_test,labels_test)

print "DT:"
from sklearn import tree
clf2 = tree.DecisionTreeClassifier()
clf2.fit(features_train,labels_train)
print clf2.score(features_test,labels_test)

print "SVM :"
from sklearn.svm import SVC
clf3 = SVC()
clf3.fit(features_train,labels_train)
print clf3.score(features_test,labels_test)

print "Random Forest: "
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(max_depth=2, random_state=0)
clf4.fit(features_train,labels_train)
print clf4.score(features_test,labels_test)

print "ADABOOST: "
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
clf5.fit(features_train,labels_train)
print clf5.score(features_test,labels_test)

print "ExtraTreesClassifier :"
from sklearn.ensemble import ExtraTreesClassifier
clf6=ExtraTreesClassifier(n_estimators=30, min_samples_split=35,random_state=0)
clf6.fit(features_train,labels_train)
print clf6.score(features_test,labels_test)

print "Logistic Regression:"
from sklearn import linear_model
clf7 = linear_model.LogisticRegression(C=1e5,solver='lbfgs',max_iter=1000,multi_class='multinomial')
clf7.fit(features_train,labels_train)
print clf7.score(features_test,labels_test)

print "Neural Network: "

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

from sklearn.neural_network import MLPClassifier
clf8 = MLPClassifier(max_iter=2000, activation='tanh', solver='adam')
clf8.fit(features_train,labels_train)
print clf8.score(features_test,labels_test)

test = pd.read_csv("./test.csv")
test_fea = test[used_features]
aid = test["ID"]
res = clf7.predict(test_fea)

df = pd.DataFrame(data={"Room": res,"Id": aid})
df.to_csv("./file4.csv", sep=',',index=False)