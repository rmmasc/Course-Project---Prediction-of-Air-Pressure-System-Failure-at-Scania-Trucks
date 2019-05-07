# -*- coding: utf-8 -*-
"""
Name: Royston Marian Mascarenhas
USC email: rmascare@usc.edu
EE559 Final Project
Spring 2019

@author: royma
"""

#import Preprocess
from util import * #contains all imports
import pickle

#in case imports are not getting imported
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from fancyimpute import KNN    
import math
from tqdm import tqdm_notebook as tqdm
from imblearn.over_sampling import SMOTE
import time
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from pycm import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
import sklearn.metrics as skm
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
'''


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


storeflag = False
histflag = False
pcaflag = True #always true

if histflag==False:
    
    #preprocessing training data
    #read the file and replace missing values with NaN
    data = Preprocess(filename="aps_failure_training_set_SMALLER.csv",missflag=True,missing_values=["na"])
    #label encode
    data.label_splitter(x=0,splitflag=True)
    data.label_encode()
    droplst = data.feature_scraper(0.06,False)
    #imputation
    data.impute_means_fit_trans()
    #resample using smote
    data.feature_std()
    tempdata = data.data
    data.resample_smote()
    
    #preprocessing test data
    data1 = Preprocess(filename="aps_failure_test_set.csv",missflag=True,missing_values=["na"])
    data1.label_splitter(x=0,splitflag=True)
    data1.label_encode()
    data1.data.drop(data1.data.columns[droplst], axis = 1, inplace = True)
    data1.nf = data1.data.shape[1]
    data1.data = data.impute_means_transform(data1.data)
    data1.data=data.feature_std_transform(data1.data)

    #storing imbalanced training data
    train_data = tempdata
    gg = Preprocess(filename="aps_failure_training_set_SMALLER.csv",missflag=True,missing_values=["na"])
    gg.label_splitter(x=0,splitflag=True)
    gg.label_encode()
    train_labels = gg.tr_labels

    tr_labels = data.tr_labels
    tr_data = data.data
    test_labels = data1.tr_labels
    test_data = data1.data
    
    pickle_out = open("trainingFull.pickle","wb")
    pickle.dump(tr_data, pickle_out)
    pickle_out.close()
    
    pickle_out = open("trainingFullLabels.pickle","wb")
    pickle.dump(tr_labels, pickle_out)
    pickle_out.close()
    
    pickle_out = open("trainingPredict.pickle","wb")
    pickle.dump(train_data, pickle_out)
    pickle_out.close()
    
    pickle_out = open("trainingPredictLabels.pickle","wb")
    pickle.dump(train_labels, pickle_out)
    pickle_out.close()
    
    pickle_out = open("testData.pickle","wb")
    pickle.dump(test_data, pickle_out)
    pickle_out.close()
    
    pickle_out = open("testDataLabels.pickle","wb")
    pickle.dump(test_labels, pickle_out)
    pickle_out.close()
        
else:
    
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

if pcaflag == True:
    datobj = Transform(tr_data,tr_labels,test_data,test_labels)
    pcadata = datobj.perfPCA(thresh=0.5,ncflag=False,lowvar=0.85,upvar=0.9)
    pcatest = datobj.pca_transform(test_data)
    
    pickle_out = open("pcadata.pickle","wb")
    pickle.dump(pcadata, pickle_out)
    pickle_out.close()
    
    pickle_out = open("pcatest.pickle","wb")
    pickle.dump(pcatest, pickle_out)
    pickle_out.close()
    
'''
pickle_in = open("pcadata.pickle","rb")
pcadata = pickle.load(pickle_in)

pickle_in = open("pcatest.pickle","rb")
pcatest = pickle.load(pickle_in)

'''

#Classifier without PCA

print("--------Classifier without PCA-----------------")
print("Choose your classifier")
print("1. Naive Bayes")
print("2. SVM")
print("3. MLP")
print("4. KNN")
print("5. Random Forests")

choice = input("Enter here: Integer only ")

while True:
    try:
       choice = int(choice)
    except ValueError:
       print("Please ascertain correct number and dial again!")
       continue
    else:
        break

model = classifier(tr_data,tr_labels,test_data,test_labels)


if(choice==1):
    model.perfGNB()
elif(choice==2):
    model.perfSVM()
elif(choice==3):
    model.perfMLP()
elif(choice==4):
    model.perfKNN()
elif(choice==5):
    model.perfRF()
    
model.accuracies()
train_pred_model = model.get_pred(train_data)
test_pred_model,prob_model = model.get_pred(test_data,probflag=True)
trainCost = model.cost(train_labels,train_pred_model,extflag=True)
testCost = model.cost(test_labels,test_pred_model,extflag=True)
model.confusion_matrix(labels=test_labels,predictions=test_pred_model,extflag=True)
model.c_report(labels=test_labels,predictions=test_pred_model,extflag=True)
neg = model.add_stat(labels=test_labels,predictions=test_pred_model,extflag=True)
model.draw_roc(test_data=test_data,test_labels=test_labels,extflag=False)
model.draw_prc(test_data=test_data,test_labels=test_labels,extflag=False)

print("************************************")
print("PCA + MODEL SELECTED")
print("************************************")


pcamodel = classifier(pcadata,tr_labels,pcatest,test_labels)

if(choice==1):
    pcamodel.perfGNB()
elif(choice==2):
    pcamodel.perfSVM()
elif(choice==3):
    pcamodel.perfMLP()
elif(choice==4):
    pcamodel.perfKNN()
elif(choice==5):
    pcamodel.perfRF()

    
pcamodel.accuracies()
test_pred_pcamodel,prob_pcamodel = pcamodel.get_pred(pcatest,probflag=True)
testCost = pcamodel.cost(test_labels,test_pred_pcamodel,extflag=True)
pcamodel.confusion_matrix(labels=test_labels,predictions=test_pred_pcamodel,extflag=True)
pcamodel.c_report(labels=test_labels,predictions=test_pred_pcamodel,extflag=True)
neg = pcamodel.add_stat(labels=test_labels,predictions=test_pred_pcamodel,extflag=True)

print("************************************")
print("CROSS VALIDATION FOR AVERAGE COST AND ACCURACY")
print("************************************")

skf = StratifiedKFold(n_splits=3)

X = np.array(pcadata.copy())
y = np.array(tr_labels.copy())
skf = StratifiedKFold(n_splits=3, shuffle=True)
acc_list = []
tacc_list = []
trcost_list = []
testcost_list = []
i = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    temp = classifier(X_train, y_train,X_test, y_test)
    if(choice==1):
        temp.perfGNB()
    elif(choice==2):
        temp.perfSVM()
    elif(choice==3):
        temp.perfMLP(layer_sizes = (100,70))
    elif(choice==4):
        temp.perfKNN()
    elif(choice==5):
        temp.perfRF()    
    tracc,tesacc = temp.accuracies()
    train_pred_w1= temp.get_pred(X_train)
    test_pred_w = temp.get_pred(X_test)
    tCost_w = temp.cost(y_train,train_pred_w1,extflag=True)
    testCost_w = temp.cost(y_test,test_pred_w,extflag=True)
    print("------------Split {0} ---------".format(i))
    print("Train Acc: {0}".format(tracc))
    print("Test Acc: {0}".format(tracc))
    print("Cost on training set: %.3f"%tCost_w)
    print("Cost on test set: %.3f"%testCost_w)
    print("-----------------------------------")
    acc_list.append(tracc)
    tacc_list.append(tesacc)
    trcost_list.append(tCost_w)
    testcost_list.append(testCost_w)
    acc_list_m = np.mean(acc_list)
    tacc_list_m = np.mean(tacc_list)
    trcost_list_m = np.mean(trcost_list)
    testcost_list_m = np.mean(testcost_list)
    i+=1
print("***********FINAL RESULTS***************")
print("Average training Accuracy is %.3f"%acc_list_m)
print("Average testing Accuracy is %.3f"%tacc_list_m)
print("Average training cost is %.3f"%trcost_list_m)
print("Average testing cost is %.3f"%testcost_list_m)
print("****************************************")


#Cross Validation for model selection

if (choice == 2):

    print("*********************MODEL SELECTION************************")    
    skf = StratifiedKFold(n_splits=3, shuffle=False)
        
    Clst = [1,50,100]
    X = np.array(train_data.copy())
    y = np.array(train_labels.copy())
    f1F = []
    recallF = []
    auc1F = []
    costF = []
    neglF = []
    i = 1
    for c_index in range(0,3):
        f1_lst = []
        recall_lst = []
        auc1_lst = []
        cost_lst = []
        negl_lst = []
        #print("1")
        for train_index, test_index in skf.split(X, y):
            Cpar = Clst[c_index]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #print("2")
            print("Cpar"+str(Cpar))
            temp = classifier(X_train, y_train,X_test, y_test)
            temp.perfSVM(kernel1="rbf",Cparam=Cpar)
            #temp.perfGNB()
            #print("Classification done")
            test_pred_w,probs = temp.get_pred(X_test,probflag=True)
            f1, recall, auc1, cost, negl = validation_metrics(labels=y_test,preds=test_pred_w,prob_list=probs)
            #print("4")
            f1_lst.append(f1)
            recall_lst.append(recall)
            auc1_lst.append(auc1)
            cost_lst.append(cost)        
            negl_lst.append(negl)        
        f1F.append(np.mean(f1_lst))
        recallF.append(np.mean(recall_lst))
        auc1F.append(np.mean(auc1_lst))
        costF.append(np.mean(cost_lst))
        neglF.append(np.mean(negl_lst))
    
        
        print("------------Parameter {0} ---------".format(i))
        print("C = {0}".format(Cpar))
        print("F1 of positive class : {0}".format(f1F[i-1]))
        print("Recall of positive class: {0}".format(recallF[i-1]))
        print("AUC of positive class: {0}".format(auc1F[i-1]))
        print("Cost of PRC Curve of positive class: {0}".format(costF[i-1]))
        print("Negligence: {0}".format(neglF[i-1]))
        print("-----------------------------------")
        
        i+=1
        
    print("***********FINAL RESULTS***************")
    print("Average F1 of positive class is %.3f"%np.mean(f1F))
    print("Average Recall of positive class %.3f"%np.mean(recallF))
    print("Average AUC is %.3f"%np.mean(auc1F))
    print("Average cost is %.3f"%np.mean(costF))
    print("Average negligence is %.3f"%np.mean(neglF))
    
    print("****************************************")
    
    bestF1 = np.argmax(f1F)
    bestAUC = np.argmax(auc1F)
    bestRecall = np.argmax(recallF)
    bestCost = np.argmin(costF)
    bestNegl = np.argmin(neglF)
    print("According to F1, best value is {0}".format(Clst[bestF1]))
    print("According to AUC, best value is {0}".format(Clst[bestAUC]))
    print("According to Recall, best value is {0}".format(Clst[bestRecall]))
    print("According to Cost, best value is {0}".format(Clst[bestCost]))
    print("According to Negligence, best value is {0}".format(Clst[bestNegl]))
    
elif choice==3:
    print("*********************MODEL SELECTION************************")    
    skf = StratifiedKFold(n_splits=3, shuffle=False)
    
    Clst = [(100,),(700,),(100,80,24),(600,350,86)]
    X = np.array(pcadata.copy())
    y = np.array(tr_labels.copy())
    f1F = []
    recallF = []
    auc1F = []
    costF = []
    neglF = []
    i = 1
    for c_index in range(0,4):
        f1_lst = []
        recall_lst = []
        auc1_lst = []
        cost_lst = []
        negl_lst = []
        #print("1")
        for train_index, test_index in skf.split(X, y):
            Cpar = Clst[c_index]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #print("2")
            temp = classifier(X_train, y_train,X_test, y_test)
            temp.perfMLP(layer_sizes=Cpar)
            #temp.perfGNB()
            #print("Classification done")
            test_pred_w,probs = temp.get_pred(X_test,probflag=True)
            f1, recall, auc1, cost, negl = validation_metrics(labels=y_test,preds=test_pred_w,prob_list=probs)
            #print("4")
            f1_lst.append(f1)
            recall_lst.append(recall)
            auc1_lst.append(auc1)
            cost_lst.append(cost)        
            negl_lst.append(negl)        
        f1F.append(np.mean(f1_lst))
        recallF.append(np.mean(recall_lst))
        auc1F.append(np.mean(auc1_lst))
        costF.append(np.mean(cost_lst))
        neglF.append(np.mean(negl_lst))
    
        
        print("------------Parameter {0} ---------".format(i))
        print("C = {0}".format(Cpar))
        print("F1 of positive class : {0}".format(f1F[i-1]))
        print("Recall of positive class: {0}".format(recallF[i-1]))
        print("AUC of positive class: {0}".format(auc1F[i-1]))
        print("Cost of PRC Curve of positive class: {0}".format(costF[i-1]))
        print("Negligence: {0}".format(neglF[i-1]))
        print("-----------------------------------")
        
        i+=1
        
    print("***********FINAL RESULTS***************")
    print("Average F1 of positive class is %.3f"%np.mean(f1F))
    print("Average Recall of positive class %.3f"%np.mean(recallF))
    print("Average AUC is %.3f"%np.mean(auc1F))
    print("Average cost is %.3f"%np.mean(costF))
    print("Average negligence is %.3f"%np.mean(neglF))
    
    print("****************************************")
    
    bestF1 = np.argmax(f1F)
    bestAUC = np.argmax(auc1F)
    bestRecall = np.argmax(recallF)
    bestCost = np.argmin(costF)
    bestNegl = np.argmin(neglF)
    print("According to F1, best value is {0}".format(Clst[bestF1]))
    print("According to AUC, best value is {0}".format(Clst[bestAUC]))
    print("According to Recall, best value is {0}".format(Clst[bestRecall]))
    print("According to Cost, best value is {0}".format(Clst[bestCost]))
    print("According to Negligence, best value is {0}".format(Clst[bestNegl]))
        
elif (choice==4):
    print("*********************MODEL SELECTION************************")    
    skf = StratifiedKFold(n_splits=3, shuffle=False)
    
    Clst = [3,10,20]
    X = np.array(pcadata.copy())
    y = np.array(tr_labels.copy())
    f1F = []
    recallF = []
    auc1F = []
    costF = []
    neglF = []
    i = 1
    for c_index in range(0,3):
        f1_lst = []
        recall_lst = []
        auc1_lst = []
        cost_lst = []
        negl_lst = []
        #print("1")
        for train_index, test_index in skf.split(X, y):
            Cpar = Clst[c_index]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #print("2")
            temp = classifier(X_train, y_train,X_test, y_test)
            temp.perfKNN(nbrs=Cpar)
            #temp.perfGNB()
            #print("Classification done")
            test_pred_w,probs = temp.get_pred(X_test,probflag=True)
            f1, recall, auc1, cost, negl = validation_metrics(labels=y_test,preds=test_pred_w,prob_list=probs)
            #print("4")
            f1_lst.append(f1)
            recall_lst.append(recall)
            auc1_lst.append(auc1)
            cost_lst.append(cost)        
            negl_lst.append(negl)        
        f1F.append(np.mean(f1_lst))
        recallF.append(np.mean(recall_lst))
        auc1F.append(np.mean(auc1_lst))
        costF.append(np.mean(cost_lst))
        neglF.append(np.mean(negl_lst))
    
        
        print("------------Parameter {0} ---------".format(i))
        print("C = {0}".format(Cpar))
        print("F1 of positive class : {0}".format(f1F[i-1]))
        print("Recall of positive class: {0}".format(recallF[i-1]))
        print("AUC of positive class: {0}".format(auc1F[i-1]))
        print("Cost of PRC Curve of positive class: {0}".format(costF[i-1]))
        print("Negligence: {0}".format(neglF[i-1]))
        print("-----------------------------------")
        
        i+=1
        
    print("***********FINAL RESULTS***************")
    print("Average F1 of positive class is %.3f"%np.mean(f1F))
    print("Average Recall of positive class %.3f"%np.mean(recallF))
    print("Average AUC is %.3f"%np.mean(auc1F))
    print("Average cost is %.3f"%np.mean(costF))
    print("Average negligence is %.3f"%np.mean(neglF))
    
    print("****************************************")
    
    bestF1 = np.argmax(f1F)
    bestAUC = np.argmax(auc1F)
    bestRecall = np.argmax(recallF)
    bestCost = np.argmin(costF)
    bestNegl = np.argmin(neglF)
    print("According to F1, best value is {0}".format(Clst[bestF1]))
    print("According to AUC, best value is {0}".format(Clst[bestAUC]))
    print("According to Recall, best value is {0}".format(Clst[bestRecall]))
    print("According to Cost, best value is {0}".format(Clst[bestCost]))
    print("According to Negligence, best value is {0}".format(Clst[bestNegl]))
    
elif(choice==5):
    print("*********************MODEL SELECTION************************")    
    skf = StratifiedKFold(n_splits=3, shuffle=False)
    
    Clst = [10,50,120]
    X = np.array(pcadata.copy())
    y = np.array(tr_labels.copy())
    f1F = []
    recallF = []
    auc1F = []
    costF = []
    neglF = []
    i = 1
    for c_index in range(0,3):
        f1_lst = []
        recall_lst = []
        auc1_lst = []
        cost_lst = []
        negl_lst = []
        #print("1")
        for train_index, test_index in skf.split(X, y):
            Cpar = Clst[c_index]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #print("2")
            temp = classifier(X_train, y_train,X_test, y_test)
            temp.perfRF(est=Cpar)
            #temp.perfGNB()
            #print("Classification done")
            test_pred_w,probs = temp.get_pred(X_test,probflag=True)
            f1, recall, auc1, cost, negl = validation_metrics(labels=y_test,preds=test_pred_w,prob_list=probs)
            #print("4")
            f1_lst.append(f1)
            recall_lst.append(recall)
            auc1_lst.append(auc1)
            cost_lst.append(cost)        
            negl_lst.append(negl)        
        f1F.append(np.mean(f1_lst))
        recallF.append(np.mean(recall_lst))
        auc1F.append(np.mean(auc1_lst))
        costF.append(np.mean(cost_lst))
        neglF.append(np.mean(negl_lst))
    
        
        print("------------Parameter {0} ---------".format(i))
        print("C = {0}".format(Cpar))
        print("F1 of positive class : {0}".format(f1F[i-1]))
        print("Recall of positive class: {0}".format(recallF[i-1]))
        print("AUC of positive class: {0}".format(auc1F[i-1]))
        print("Cost of PRC Curve of positive class: {0}".format(costF[i-1]))
        print("Negligence: {0}".format(neglF[i-1]))
        print("-----------------------------------")
        
        i+=1
        
    print("***********FINAL RESULTS***************")
    print("Average F1 of positive class is %.3f"%np.mean(f1F))
    print("Average Recall of positive class %.3f"%np.mean(recallF))
    print("Average AUC is %.3f"%np.mean(auc1F))
    print("Average cost is %.3f"%np.mean(costF))
    print("Average negligence is %.3f"%np.mean(neglF))
    
    print("****************************************")






