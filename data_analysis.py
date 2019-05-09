# -*- coding: utf-8 -*-
"""
Name: Royston Marian Mascarenhas
USC email: rmascare@usc.edu
EE559 Final Project
Spring 2019

@author: royma
"""

from util import * #contains all imports
import pickle
from sklearn.feature_selection import RFE

#***************SELF WRITTEN CLASSES*********************
from Transform import Transform
from classifier import classifier
from Preprocess import Preprocess
#********************************************************
    
    
def validation_metrics(labels,preds,prob_list):
    print(np.shape(labels))
    print(np.shape(preds))
    conf_mat = skm.confusion_matrix(labels,preds)
    FN = conf_mat[1,0]
    FP = conf_mat[0,1]
    TN = conf_mat[0,0]
    TP = conf_mat[1,1]
    print("False neg:"+str(FN))
    cost = FN*500 + FP*10
    negligence = (FN/(TP+FN)) * 100
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1  = 2 * ((precision*recall)/(precision+recall))
    prob_list = prob_list[:,1].copy()
    precision1, recall1, thresh = precision_recall_curve(labels,prob_list)
    auc1 = auc(recall1, precision1)
    return f1, recall, auc1, cost, negligence



pickle_in = open("trainingFull.pickle","rb")
tr_data = pickle.load(pickle_in)
    
pickle_in = open("trainingFullLabels.pickle","rb")
tr_labels = pickle.load(pickle_in)
    
pickle_in = open("trainingPredict.pickle","rb")
train_data = pickle.load(pickle_in)
    
pickle_in = open("trainingPredictLabels.pickle","rb")
train_labels = pickle.load(pickle_in)
    
pickle_in = open("testData.pickle","rb")
test_data = pickle.load(pickle_in)

pickle_in = open("testDataLabels.pickle","rb")
test_labels = pickle.load(pickle_in)

pickle_in = open("pcadata.pickle","rb")
pcadata = pickle.load(pickle_in)

pickle_in = open("pcatest.pickle","rb")
pcatest = pickle.load(pickle_in)


adata = tr_data.iloc[:,0:49].copy()
bdata = tr_data.iloc[:,49:74].copy()
cdata = tr_data.iloc[:,74:104].copy()
ddata = tr_data.iloc[:,104:117].copy()
edata = tr_data.iloc[:,117:].copy()

adataT = test_data.iloc[:,0:49].copy()
bdataT = test_data.iloc[:,49:74].copy()
cdataT = test_data.iloc[:,74:104].copy()
ddataT = test_data.iloc[:,104:117].copy()
edataT = test_data.iloc[:,117:].copy()

print("********************************************")
print("Analyzing performance of ax_xxx machine parts")
print("********************************************")
temp = classifier(adata,tr_labels,adataT,test_labels)
temp.perfRF()
temp.accuracies()
train_pred_temp = temp.get_pred(adata)
test_pred_temp,prob_temp = temp.get_pred(adataT,probflag=True)
trainCost = temp.cost(tr_labels,train_pred_temp,extflag=True)
testCost = temp.cost(test_labels,test_pred_temp,extflag=True)
conf_mat = skm.confusion_matrix(test_labels,test_pred_temp)
plt.figure(1)
sns.heatmap(conf_mat, annot=True,fmt='g')


print("********************************************")
print("Analyzing performance of bx_xxx machine parts")
print("********************************************")
temp = classifier(bdata,tr_labels,bdataT,test_labels)
temp.perfRF()
temp.accuracies()
train_pred_temp = temp.get_pred(bdata)
test_pred_temp,prob_temp = temp.get_pred(bdataT,probflag=True)
trainCost = temp.cost(tr_labels,train_pred_temp,extflag=True)
testCost = temp.cost(test_labels,test_pred_temp,extflag=True)
conf_mat = skm.confusion_matrix(test_labels,test_pred_temp)
plt.figure(2)
sns.heatmap(conf_mat, annot=True,fmt='g')

print("********************************************")
print("Analyzing performance of ac_xxx machine parts")
print("********************************************")
temp = classifier(cdata,tr_labels,cdataT,test_labels)
temp.perfRF()
temp.accuracies()
train_pred_temp = temp.get_pred(cdata)
test_pred_temp,prob_temp = temp.get_pred(cdataT,probflag=True)
trainCost = temp.cost(tr_labels,train_pred_temp,extflag=True)
testCost = temp.cost(test_labels,test_pred_temp,extflag=True)
conf_mat = skm.confusion_matrix(test_labels,test_pred_temp)
plt.figure(3)
sns.heatmap(conf_mat, annot=True,fmt='g')

print("********************************************")
print("Analyzing performance of dx_xxx machine parts")
print("********************************************")
temp = classifier(ddata,tr_labels,ddataT,test_labels)
temp.perfRF()
temp.accuracies()
train_pred_temp = temp.get_pred(ddata)
test_pred_temp,prob_temp = temp.get_pred(ddataT,probflag=True)
trainCost = temp.cost(tr_labels,train_pred_temp,extflag=True)
testCost = temp.cost(test_labels,test_pred_temp,extflag=True)
conf_mat = skm.confusion_matrix(test_labels,test_pred_temp)
plt.figure(4)
sns.heatmap(conf_mat, annot=True,fmt='g')

print("********************************************")
print("Analyzing performance of ex_xxx machine parts")
print("********************************************")
temp = classifier(edata,tr_labels,edataT,test_labels)
temp.perfRF()
temp.accuracies()
train_pred_temp = temp.get_pred(edata)
test_pred_temp,prob_temp = temp.get_pred(edataT,probflag=True)
trainCost = temp.cost(tr_labels,train_pred_temp,extflag=True)
testCost = temp.cost(test_labels,test_pred_temp,extflag=True)
plt.figure(5)
conf_mat3 = skm.confusion_matrix(test_labels,test_pred_temp)
sns.heatmap(conf_mat3, annot=True,fmt='g')

col_names = ['bt_000','am_0', 'aa_000', 'ag_000', 'cs_008']

fig, ax = plt.subplots(len(col_names), figsize=(12,12))

for i, col_val in enumerate(col_names):

    sns.distplot(tr_data[col_val], hist=True, ax=ax[i])
    ax[i].set_title('Freq disribution '+col_val)
    ax[i].set_xlabel(col_val)
    ax[i].set_ylabel('Count')

plt.show()

#outlier detection plots

print("OUTLIER DETECTION PLOTS")
data = Preprocess(filename="aps_failure_training_set_SMALLER.csv",missflag=True,missing_values=["na"])

data.label_splitter(x=0,splitflag=True)
data.label_encode()
data.impute_means_fit_trans()
gd = data.data
col_names = ['ab_000','ad_000']

fig, ax = plt.subplots(len(col_names), figsize=(8,10))

for i, col_val in enumerate(col_names):

    sns.boxplot(y=gd[col_val], ax=ax[i])
    ax[i].set_title('Box plot - {}'.format(col_val), fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)

plt.show()

#correlation plots

print("CORRELATION PLOTS")
f, ax = plt.subplots(figsize=(10, 10))
corr = tr_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

f, ax = plt.subplots(figsize=(10, 8))
corr = pcadata.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


