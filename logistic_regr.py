import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import log_loss

#loading the data from csv file
churn_df = pd.read_csv("ChurnData.csv")
print(churn_df.head())

#data preprocessing and selection and changing the target data type to be int as required by skitlearn
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

#lets define our feature and target variables
X = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']].values
print(X[0:5])
y=churn_df[['churn']].values
print(y[0:5])

#normalizing the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

#spliting of dataset into test and train
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

#fitting the data into the model
'''
We can use different numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers.
C parameter indicates inverse of regularization strength which must be a positive float.
Smaller values specify stronger regularization.
'''
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
print(LR)

#Predicting the test dataset
yhat = LR.predict(X_test)
print(yhat)
'''
predict_proba returns estimates for all classes, ordered by the label of classes.
So, the first column is the probability of class 1, P(Y=1|X),
and second column is probability of class 0, P(Y=0|X)
'''
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

'''EVALUATION'''

#jacard index
print('The jacard index ',jaccard_similarity_score(y_test, yhat))

#confusion matrix
'''
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

'''
# Plot non-normalized confusion matrix
plt.figure()
print(plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix'))
plt.show()
'''
#the classifiaction report
print ('The classifiaction report \n',classification_report(y_test, yhat))

#logloss
print('The log loss',log_loss( y_test, yhat_prob))
