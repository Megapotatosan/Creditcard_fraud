
# coding: utf-8

# In[87]:


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('matplotlib inline')


# In[88]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[89]:


data = pd.read_csv("E:/Google Drive/Ustudy/research/creditcard.csv")
data.head()
columns=data.columns
features_columns=columns.delete(len(columns)-1)

features=data[features_columns]
labels=data['Class']
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']


# In[90]:


# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            labels, 
                                                                            test_size=0.3, 
                                                                            random_state=0)
print("Number transactions train dataset: ", len(features_train))
print("Number transactions test dataset: ", len(features_test))
print("Total number of transactions: ", len(features_train)+len(features_test))


# In[97]:


oversampler=SMOTE(random_state=1)
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)


# In[98]:


clf=RandomForestClassifier(n_estimators=15,random_state=1)
clf2=RandomForestClassifier(n_estimators=30,random_state=1)
clf.fit(os_features,os_labels)
clf2.fit(whole_os_features,whole_os_labels)


# In[99]:


actual=labels_test
predictions=clf.predict(features_test)
y_pred = clf2.predict(X_test)


# In[100]:


confusion_matrix(actual,predictions)


# In[101]:


cnf_matrix = confusion_matrix(actual,predictions)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
print("Precision metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[0,1]+cnf_matrix[1,1]))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[102]:


confusion_matrix(y_test,y_pred)
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
print("Precision metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[0,1]+cnf_matrix[1,1]))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

