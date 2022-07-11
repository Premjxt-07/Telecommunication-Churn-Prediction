# DEPENDENCIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as py
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import log_loss


"%matplotlib inline"

df = pd.read_csv("ChurnData.csv")
# print(df.head())

# print(df.columns)

df_churn = df[['tenure', 'age', 'address', 'income', 'ed',
               'employ', 'equip', 'callcard', 'wireless', 'churn']].astype('int')
# print(df_churn.head())
# print(df_churn.dtypes)
# print(df_churn.shape)


""" Selection few parameters or arguments from the list of arguments """

X = np.asanyarray(df_churn[['tenure', 'age', 'address',
                            'income', 'ed', 'employ', 'equip']])
# print(X)
y = np.asanyarray(df_churn['churn'])

""" Data Preprocessing  """

X = preprocessing.StandardScaler().fit(X).transform(X)
# print(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)

""" TO find the number of rows and columns after splitting """

# print(" Train shape ", X_train.shape,y_train.shape)
# print(" Test shape ", X_test.shape,y_test.shape)

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# print(LR)

y_pred = LR.predict(X_test)
# print(y_pred)
y_prob = LR.predict_proba(X_test)

""" Jaccard index """
jaccard_index = jaccard_similarity_score(y_test, y_pred)
# print(jaccard_index)


""" CONFUSION MATRIX """


""" def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
# This function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize = True`.
"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# print(confusion_matrix(y_test, y_pred, labels=[1,0]))

cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()

print (classification_report(y_test, y_pred))

 """

""" LOG LOSS """

print(log_loss(y_test, y_prob))
