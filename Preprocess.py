# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:04:48 2019

@author: royma
"""

from util import *


class Preprocess ():
    '''
    Methods:
    
    Default constructor: The default initialization is flexible. Data can be input externally, csv file can be read, 
    or optionally, csv file can be read with missing values replaced by NaN. Note that the missing values will have to 
    be passed as a list type in the third case. To use the preprocess class, the data has to be made a property of its
    instance. 
    
    feature_scraper: Drops features with missing values beyond the quantity defined by trust_threshold. If the trust 
    threshold is 0.1, it implies that features which have missing values for more than 10% of the number of data points 
    will be discarded.
    '''
    def __init__(self,filename=" ",extreadflag=False,ext_data=[],missflag = False,missing_values=[]):
        if extreadflag == True:
            self.data = ext_data
        if missflag == True:
            self.data = pd.read_csv(filename, na_values = missing_values)
        else:
            self.data = pd.read_csv(filename)
        self.n = self.data.shape[0] #number of data points
        self.nf = self.data.shape[1] #number of features
        self.history = 0
        self.historydf = pd.DataFrame()
        self.historyexp = []
        
    def label_splitter(self,x,splitflag=True,labels = []):
        if (splitflag == True):
            self.tr_labels = self.data.iloc[:,x]
            self.data.drop(self.data.columns[x], axis = 1, inplace = True)
            self.nf = self.data.shape[1]
        else:
            self.tr_labels = labels
        #print(self.tr_labels)
    #def reinit(self,histindex):
        #self.data = self.historydf[histindex]
    
    def label_encode (self):
        u_label = len(np.unique(self.tr_labels))
        lst = np.zeros(self.n)
        pre = np.unique(self.tr_labels)
        post = np.zeros(u_label)
        for i in range(0,u_label):
            post[i] = i
            print(str(pre[i])+"is replaced by "+str(post[i]))
        for i in range(0,len(self.tr_labels)):
            if self.tr_labels[i] == pre[0]:
                lst[i] = post[0]
            elif self.tr_labels[i] == pre[1]:
                lst[i] = post[1]
        self.tr_labels = pd.Series(lst)
        
    def feature_scraper(self,trust_threshold,trialflag=False,retflag=True):
        discard = int(np.ceil(trust_threshold * self.n))
        print("Threshold is "+str(discard))
        drop_lst = []
        for i in range(0,self.nf):
            temp = self.data.iloc[:,i].isnull().sum()
            if temp > discard:
                drop_lst.append(i)
        print(str(len(drop_lst))+" features will be discarded from "+str(self.nf))
        if (trialflag==True):
            prflag = input("Proceed? Y/N")
            if (prflag.lower() == "y"):
                self.data.drop(self.data.columns[drop_lst], axis = 1, inplace = True)
                self.nf = self.data.shape[1]
                print("Number of features post process: "+str(self.nf))
            else:
                print("Preprocess cancelled. Please recall method")
        else:
            self.data.drop(self.data.columns[drop_lst], axis = 1, inplace = True)
            self.nf = self.data.shape[1]
            print("Number of features post process: "+str(self.nf))
        if(retflag==True):
            return drop_lst
        
    def impute_means_fit_trans(self):
        self.mean_lst = []
        for i in tqdm(range(0,self.nf)):
            lst = [self.data.iloc[x,i] for x in range(0,self.n) if (math.isnan(self.data.iloc[x,i])== False)]
            lstmean = np.mean(lst)
            self.mean_lst.append(lstmean)
            for j in range(0,self.n):
                if (math.isnan(self.data.iloc[j,i]) == True):
                    self.data.iloc[j,i] = lstmean
    
    def impute_means_transform(self,data):
        n,nf=data.shape
        for i in tqdm(range(0,nf)):
            for j in range(0,n):
                if (math.isnan(data.iloc[j,i]) == True):
                    data.iloc[j,i] = self.mean_lst[i]
        return data
            

    def impute_classmeans(self):
        # iterate through all rows of a feature
        #calculate the means of the feature corresponding to each class
        #replace all missing values with class wise feature mean
        for i in tqdm(range(0,self.nf)):
            pos_lst = [self.data.iloc[x,i] for x in range(0,self.n) if (math.isnan(self.data.iloc[x,i])== False and self.tr_labels[x] == 1)]
            neg_lst = [self.data.iloc[x,i] for x in range(0,self.n) if (math.isnan(self.data.iloc[x,i])== False and self.tr_labels[x] == 0)]
            pos_mean = np.mean(pos_lst)
            neg_mean = np.mean(neg_lst)
            for j in range(0,self.n):
                if (math.isnan(self.data.iloc[j,i]) == True and self.tr_labels[j] == 1):
                    self.data.iloc[j,i] = pos_mean
                elif (math.isnan(self.data.iloc[j,i]) == True and self.tr_labels[j] == 0):
                    self.data.iloc[j,i] = neg_mean
                    
                    
    def feature_std(self):
        self.col = list(self.data.columns)
        self.stdobj = StandardScaler()
        self.stdobj.fit(self.data)
        self.data = self.stdobj.transform(self.data)
        self.data = pd.DataFrame(self.data,columns = self.col)
    
    def resample_smote(self):
        col = list(self.data.columns)
        sm = SMOTE(sampling_strategy='minority')
        self.data,self.tr_labels = sm.fit_resample(self.data,self.tr_labels)
        self.data = pd.DataFrame(self.data,columns = col)
        self.n = self.data.shape[0] #number of data points
        self.nf = self.data.shape[1] #number of features
        
    def custom_std(self,doflag=False):
        self.meanlst=[]
        self.stdlst=[]
        for i in range(0,self.nf):
            fmean = np.mean(self.data.iloc[:,i])
            fstd = np.std(self.data.iloc[:,i])
            self.meanlst.append(fmean)
            self.stdlst.append(fstd)
            if(doflag==True):
                (self.data.iloc[:,i]-fmean)/fstd
            
    def ret_params(self):
        return self.data, self,tr_labels,self.meanlst,self.stdlst

    def feature_std_transform(self,data):
        return (pd.DataFrame(self.stdobj.transform(data),columns = self.col))
    '''def history (self,old_data,comment):
        self.historydf[self.hist] = old_data 
        self.history
        self.history = self.history + 1
        
    def print_history(self,)'''
        
        
        
        
    