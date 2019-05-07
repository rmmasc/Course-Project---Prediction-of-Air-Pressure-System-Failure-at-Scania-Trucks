# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:06:23 2019

@author: royma
"""

from util import *

class classifier():
    def __init__(self,data,labels,tata,tabels):
        self.data = data
        self.labels = labels
        self.tata = tata
        self.tabels = tabels     
        self.pred = []
        self.tpred = []
        self.n = self.data.shape[0]
        self.nf = self.data.shape[1]
        self.tn = self.tata.shape[0]
        self.tnf = self.tata.shape[1]
        
    def perfGNB(self):
        start = time.time()
        self.model = GaussianNB()
        self.model.fit(self.data,self.labels)
        self.pred = self.model.predict(self.data)
        self.tpred = self.model.predict(self.tata)
        #print(self.tpred)
        self.ind = "Gaussian Naive Bayes"
        end = time.time()
        print("The time taken for "+self.ind+" execution is "+ str(end - start))
        return self.pred,self.tpred
        
    def perfSVM(self,kernel1='rbf',Cparam=1.0,gam='scale',valflag=False):
        if(valflag==False):
            start = time.time()
            self.model = svm.SVC(C=Cparam,kernel=kernel1,gamma=gam,probability=True)
            self.model.fit(self.data,self.labels)
            self.pred = self.model.predict(self.data)
            self.tpred = self.model.predict(self.tata)
            self.ind = "SVM"
            end = time.time()
            print("The time taken for "+self.ind+" with "+kernel1+" is "+ str(end - start))
        if (valflag==True):
            start = time.time()
            params = {}
            candidates = {'C': [1, 10, 100], 'kernel': ['linear','rbf']}
            self.model = svm.SVC(kernel=kernel1,gamma='scale',probability=True)
            clf = GridSearchCV(self.model, candidates, cv=5)
            clf.fit(self.data,self.labels)
            print('Best C:',clf.best_estimator_.C) 
            print('Best Kernel:',clf.best_estimator_.kernel)
            print('Best Gamma:',clf.best_estimator_.gamma)
            self.pred = self.model.predict(self.data)
            self.tpred = self.model.predict(self.tata)
            self.ind = "SVM"
            end = time.time()
            print("The time taken for "+self.ind+" with "+kernel1+" is "+ str(end - start))
            
            
    def perfMLP(self,layer_sizes=(100,),solver1='sgd',lr='constant',epochs=1000,lr_init=0.001):
        start = time.time()
        self.model = MLPClassifier(hidden_layer_sizes=layer_sizes, activation='relu', solver=solver1, 
                                                     learning_rate=lr, max_iter=epochs, shuffle=True, 
                                                     early_stopping=True,learning_rate_init=lr_init)
        self.model.fit(self.data,self.labels)
        self.pred = self.model.predict(self.data)
        self.tpred = self.model.predict(self.tata)
        self.ind = "MLP"
        end = time.time()
        print("The time taken for "+self.ind+" with lr= "+str(lr)+" with solver= "+str(solver1)+" and epochs = "
              +str(epochs)+" is "+ str(end - start))
        
    
    def perfKNN(self,nbrs=3):
        start = time.time()
        self.model = KNeighborsClassifier(n_neighbors=nbrs)
        self.model.fit(self.data,self.labels)
        self.pred = self.model.predict(self.data)
        self.tpred = self.model.predict(self.tata)
        end = time.time()
        self.ind = "KNN"
        end = time.time()
        print("The time taken for "+self.ind+" with neighbors= "+str(nbrs)+" is "+ str(end - start))
    
    def perfRF(self,est=10):
        start = time.time()
        self.model = RandomForestClassifier(n_estimators=est, max_depth = 2,random_state=-0)
        self.model.fit(self.data,self.labels)
        self.pred = self.model.predict(self.data)
        self.tpred = self.model.predict(self.tata)
        end = time.time()
        self.ind = "Random Forests"
        end = time.time()
        print("The time taken for "+self.ind+" with number of estimators= "+str(est)+" is "+ str(end - start))
        
        
    def accuracies(self,labels=[],testlabels=[],tr_pred=[],test_pred=[],extflag=False):
        if (extflag == False):
            labels = self.labels
            testlabels=self.tabels
            tr_pred=self.pred
            test_pred=self.tpred
        count = 0
        n = labels.shape[0]
        #print(tr_pred)
        #print(labels.shape)
        #print(np.shape(tr_pred))
        #print(np.shape(test_labels))
        #print(np.shape(test_pred))
        for i in range(0,n):
            if (tr_pred[i]==labels[i]):
                count = count + 1
        train_acc = (count/n)*100
        print(count)
        count = 0
        tn = testlabels.shape[0]
        for i in range(0,tn):
            if (test_pred[i]==testlabels[i]):
                count = count + 1
        test_acc = (count/tn)*100
        print(count)
        print("The training accuracy using "+self.ind+" is "+str(train_acc))
        print("The test accuracy using "+self.ind+" is "+str(test_acc))
        return train_acc,test_acc
    
    
    def confusion_matrix(self,labels=[],predictions=[],extflag=False):
        if (extflag == False):
            labels=self.tabels
            predictions=self.tpred
        self.conf_mat = skm.confusion_matrix(labels,predictions)
        sns.heatmap(self.conf_mat, annot=True,fmt='g')
        return self.conf_mat
        
    def c_report(self,labels=[],predictions=[],extflag=False):
        if (extflag == False):
            labels=self.tabels
            predictions=self.tpred
        self.rep1 = skm.classification_report(labels,predictions)
        print(self.rep1)
        
    def cost(self,labels=[],predictions=[],extflag = False):
        if (extflag == False):
            print("Using object data")
            labels=self.tabels
            predictions=self.tpred
        self.conf_mat = skm.confusion_matrix(labels,predictions)
        FN = self.conf_mat[1,0]
        FP = self.conf_mat[0,1]
        TN = self.conf_mat[0,0]
        TP = self.conf_mat[1,1]
        print("Type 1 errors : {0} and Type 2 errors: {1}".format(FP,FN))
        cost = FN*500 + FP*10
        print("The total incurred by the model is {0} USD".format(cost))        
        return cost 
    
    def add_stat(self,labels=[],predictions=[],extflag=False,verbose=True):
        if (extflag == False):
            labels=self.tabels
            predictions=self.tpred
        self.conf_mat = skm.confusion_matrix(labels,predictions)
        FN = self.conf_mat[1,0]
        FP = self.conf_mat[0,1]
        TN = self.conf_mat[0,0]
        TP = self.conf_mat[1,1]
        
        negligence = (FN/(TP + FN)) * 100
        sensitivity = TP/(TP+FN) #sensitivity of the model to the positive class (to minority APS class)
        specificity = TN/(TN+FP) 
        FPR = 1 - specificity
        PPV = TP/(TP + FP) #aps when you dont have aps
        if(verbose==True):
            print("The sensitivity/true positive rate of the model is %.3f" %(sensitivity))
            print("The specficity of the model is %.3f" %specificity)
            print("The False Positive rate is %.3f" %FPR)
            print("The positive predicate value of the model is %.3f" %PPV)
        print("The negligence of the model is %.3f"%negligence)
        return negligence 
    
    def draw_roc(self,test_data=[],test_labels=[],extflag=False):
        if (extflag == False):
            test_data=self.tata
            test_labels=self.tabels
        self.prob_list = self.model.predict_proba(test_data)
        temp = self.model.predict(test_data)
        prob_list = self.prob_list[:,1].copy()
        auc1 = roc_auc_score (test_labels,prob_list)
        f1 = f1_score(test_labels, temp)
        fpr, tpr, thresh = roc_curve(test_labels,prob_list)
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        ax.set_title('ROC/AUC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--',label='Lowest Skill Level')
        plt.plot(fpr, tpr,label='ROC curve')
        ax.set_xlabel('False Positive Rate (1-Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.legend(loc='best')
        plt.show()
        print("AUC is {0}".format(auc1))
        return self.prob_list, auc1, f1

    def draw_prc(self,test_data=[],test_labels=[],extflag=False):
        if (extflag == False):
            test_data=self.tata
            test_labels=self.tabels
        self.prob_list = self.model.predict_proba(test_data)
        prob_list = self.prob_list[:,1].copy()
        precision, recall, thresh = precision_recall_curve(test_labels,prob_list)
        temp = self.model.predict(test_data)
        auc1 = auc(recall, precision)
        score = average_precision_score(test_labels,prob_list)
        print('AUC=%.4f average precision score=%.4f' % (auc1, score))
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        ax.set_title('Precision/Recall Curve')
        plt.plot([0, 1], [0, 0], linestyle='--',label='Lowest Skill')
        plt.plot(recall, precision, label='Precision/Recall curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='best')
        plt.show()
        
    def get_pred(self,data,probflag = False):
        predictions = self.model.predict(data)
        if(probflag==True):
            prob_list = self.model.predict_proba(data) 
            return predictions, prob_list
        else:
            return predictions
    
    '''def SVM
    start = time.time()

    end = time.time()
    print(end - start)'''