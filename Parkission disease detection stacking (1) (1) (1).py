#!/usr/bin/env python
# coding: utf-8

# In[2]:

# Importing Libraries
from matplotlib import pyplot
from numpy import std
from numpy import mean
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import requests
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from IPython.display import display


# In[3]:


d = pd.read_csv("Parkinsson disease.csv")
d.tail()


# In[4]:


d.head()


# In[5]:


d.shape


# In[6]:


d.count()


# In[7]:


print('Number of Features In Dataset :', d.shape[1])
print('Number of Instances In Dataset : ', d.shape[0])


# In[8]:


# Dropping The Name Column
d.drop(['name'], axis=1, inplace=True)


# In[9]:


print('Number of Features In Dataset :', d.shape[1])
print('Number of Instances In Dataset : ', d.shape[0])


# In[10]:


d.info()


# In[11]:


d['status'] = d['status'].astype('uint8')


# In[12]:


d.info()


# In[13]:


# Exploring Imabalance In Dataset
d['status'].value_counts()


# In[14]:


# Extracting Features Into Features & Target
X = d.drop(['status'], axis=1)
y = d['status']

print('Feature (X) Shape Before Balancing :', X.shape)
print('Target (y) Shape Before Balancing :', y.shape)


# In[15]:


# Intialising SMOTE Object
sm = SMOTE(random_state=300)


# In[16]:


# Resampling Data
X, y = sm.fit_resample(X, y)


# In[17]:


print('Feature (X) Shape After Balancing :', X.shape)
print('Target (y) Shape After Balancing :', y.shape)


# In[18]:


# Scaling features between -1 and 1  for mormalization
scaler = MinMaxScaler((-1, 1))
# define X_features , Y_labels
X_features = scaler.fit_transform(X)
Y_labels = y
# splitting the dataset into traning and testing sets into 80 - 20
X_train, X_test, y_train, y_test = train_test_split(
    X_features, Y_labels, test_size=0.20, random_state=20)


# In[19]:


# SVM
# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Test Set Accuracy:", metrics.accuracy_score(y_test, y_pred))

X_pred = clf.predict(X_train)
print("Train Set Accuracy:", metrics.accuracy_score(y_train, X_pred))


# In[20]:


param_grid = {'kernel': ['linear'], 'C': [1],
              'gamma': [1]}

grid_SVC = GridSearchCV(svm.SVC(), param_grid, scoring='f1', verbose=5)
grid_SVC.fit(X_train, y_train)

# print best parameter after tuning
print("\nBest Parameters: ", grid_SVC.best_params_)

# print how our model looks after hyper-parameter tuning
print("\n", grid_SVC.best_estimator_)

predSVC = grid_SVC.predict(X_test)

# print classification report
print("\n", classification_report(y_test, predSVC))


# In[21]:


print("accuracy :", accuracy_score(y_test, predSVC))


# In[22]:


print(confusion_matrix(y_test, predSVC))


# In[23]:


plot_confusion_matrix(grid_SVC, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for SVM', y=1.1)
plt.show()


# In[24]:


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[25]:


models = dict()
models['svm'] = svm.SVC()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
scores


# In[26]:


# from matplotlib.pyplot import plt
#plt.boxplot(results, labels=names, showmeans=True)
# plt.show()
plt.boxplot(scores, showmeans=True)
plt.show()


# In[25]:


# KNN

Ks = 10
mean_acc = []
ConfustionMx = []
for n in range(2, Ks):

    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc.append(metrics.accuracy_score(y_test, yhat))
print('Neighbor Accuracy List')
print(mean_acc)


# In[27]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predKNN = knn.predict(X_test)
print(classification_report(y_test, predKNN))


# In[28]:


print("accuracy :", accuracy_score(y_test, predKNN))


# In[29]:


plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix KNN', y=1.1)
plt.show()


# In[30]:


folds = RepeatedStratifiedKFold(n_splits=3)


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[31]:


scores = evaluate_model(model, X, y)
scores
base_models = list()
base_models.append(('knn', KNeighborsClassifier(n_neighbors=5)))


# In[32]:


plt.boxplot(scores, showmeans=True)
plt.show()


# In[31]:


# RANDOM FOREST

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predRF = rfc.predict(X_test)

print(classification_report(y_test, predRF))


# In[55]:


print("accuracy :", accuracy_score(y_test, predRF))


# In[56]:


print("accuracy :", accuracy_score(y_test, predRF))


# In[57]:


plot_confusion_matrix(rfc, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.show()


# In[58]:


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[59]:


scores = evaluate_model(model, X, y)
scores
base_models = list()

base_models.append(('random_forest', RandomForestClassifier(n_estimators=3)))


# In[60]:


plt.boxplot(scores, showmeans=True)
plt.show()


# In[33]:


# LOGISTIC
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predlog = logmodel.predict(X_test)


# In[34]:


print(classification_report(y_test, predlog))
print("Confusion Matrix:")
confusion_matrix(y_test, predlog)


# In[35]:


print("accuracy :", accuracy_score(y_test, predlog))


# In[36]:


plot_confusion_matrix(logmodel, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for Logistic Regression', y=1.1)
plt.show()


# In[38]:


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[39]:


models = dict()
depth = []
for i in range(0, 1):
    clf = LogisticRegression()
    clf = clf.fit(X_train, y_train)
    depth.append((i, clf.score(X_test, y_test)))
print(depth)


def kfoldCV(f=1, k=1, model="logistic"):
    data = cross_validation_split(dataset, f)
    result = []


# In[40]:


plt.boxplot(depth, showmeans=True)
plt.show()


# In[41]:


# DECISION TREE
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predDT = clf.predict(X_test)

print(classification_report(y_test, predDT))


# In[42]:


d_classifier = DecisionTreeClassifier(max_leaf_nodes=20, random_state=2)
d_classifier.fit(X_train, y_train)


# In[43]:


y_predicted = d_classifier.predict(X_test)
print("accuracy :", accuracy_score(y_test, predDT))


# In[44]:


plot_confusion_matrix(d_classifier, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion matrix for Decision Tree', y=1.1)
plt.show()


# In[45]:


#folds = RepeatedStratifiedKFold(n_splits=3)


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[46]:


models = dict()
depth = []
for i in range(0, 1):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    depth.append((i, clf.score(X_test, y_test)))
print(depth)


def kfoldCV(f=1, k=1, model="DecisionTree Classifier"):
    data = cross_validation_split(dataset, f)
    result = []


# In[47]:


plt.boxplot(depth, showmeans=True)
plt.show()


# In[48]:


base_models = list()
base_models.append(('knn', KNeighborsClassifier(n_neighbors=3)))
base_models.append(('svm', SVC()))
meta_Learner = LogisticRegression()


# In[49]:


# define dataset
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)


# In[50]:


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[51]:


stacked_model = StackingClassifier(
    estimators=base_models, final_estimator=meta_Learner, cv=4)


# In[ ]:


models = dict()
models['knn'] = KNeighborsClassifier(n_neighbors=6)
models['svm'] = SVC()
depth = []

models['lr'] = LogisticRegression()
models['dt'] = DecisionTreeClassifier()

models['stacking'] = stacked_model
results, names = list(), list()
print('Base models individual performance')
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    if name == 'stacking':
        print('')
        print('Stacked Classifier Performance=')
    print('%s %.3f (%.3f)' % (name, mean(scores), std(scores)))


# In[ ]:


pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
