# -*- coding: utf-8 -*-
"""
Name: Royston Marian Mascarenhas
USC email: rmascare@usc.edu
EE559 Final Project
Spring 2019

@author: royma
"""
from util import *


class Transform():
    def __init__(self,data,labels,tdata,tlabels):
        self.data = data
        self.labels = labels
        self.tdata = tdata
        self.tlabels = tlabels
        self.n = self.data.shape[0]
        self.nf = self.data.shape[1]
        self.col = self.data.columns
        
    def perfPCA(self,nc=3,ncflag=True,lowvar=0.9,thresh=0.5,upvar=0.95):
        if (ncflag==True):
            self.pcaobj = PCA(n_components=nc)
            self.transdata = self.pcaobj.fit_transform(self.data)
            print("Number of principal components: "+str(self.transdata.shape[1]))
            print("Variance of each principal component: ")
            print(self.pcaobj.explained_variance_ratio_)
            result_var = sum (self.pcaobj.explained_variance_ratio_)
            print("Total variance covered by principal components: "+ str(result_var))
            self.transdata = pd.DataFrame(self.data,columns = self.col)
            return self.transdata
        else:
            endflag = 0
            varlst = []
            complst = []
            threshdata = int(np.floor(thresh*self.data.shape[1]))
            print("number of components needed "+str(threshdata))
            for ivar in tqdm(np.arange(lowvar,0.98,0.005)):
                self.pcaobj = PCA(ivar) 
                self.pcaobj.fit(self.data)
                self.transdata = self.pcaobj.transform(self.data)
                result_var = sum (self.pcaobj.explained_variance_ratio_)
                print("Total variance covered by principal components: "+ str(result_var)+" for"+
                      str(self.transdata.shape[1])+" components ")
                complst.append(self.transdata.shape[1])
                varlst.append(result_var)
                if (ivar > upvar and self.transdata.shape[1] < threshdata):
                    endflag = 1
                    break
                else:
                    print("discarded")
            if (endflag == 0 ):
                print("Best variance/component combination not found")
                self.pcaobj = PCA(lowvar) 
                self.pcaobj.fit(self.data)
                self.transdata = self.pcaobj.transform(self.data)
                #self.transdatat = self.pcaobj.transform(self.tdata)
                result_var = sum (self.pcaobj.explained_variance_ratio_)
            else:
                #self.transdatat = self.pcaobj.transform(self.tdata)
                plt.figure(1)
                plt.title("Variance v/s number of components graph")
                plt.xlabel("Number of components")
                plt.ylabel("Increasing variance")
                plt.plot(complst,varlst)
                plt.show()
            print("Final number of principal components: "+str(self.transdata.shape[1]))
            print("Variance of each principal component: ")
            print(self.pcaobj.explained_variance_ratio_)
            result_var = sum (self.pcaobj.explained_variance_ratio_)
            print("Final total variance covered by principal components: "+ str(result_var))
            self.transdata = pd.DataFrame(self.transdata)
            return self.transdata
    
    def hot_encode(self):
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(self.labels)
        enc.transform(self.labels)
    
    def pca_transform(self,data):
        return (self.pcaobj.transform(data))
    '''def visualize(self):
            pcaobj = PCA(n_components=3)
            self.transdata = pcaobj.fit_transform(self.data)
            result_var = sum (pcaobj.explained_variance_ratio_)
            print("Fidelity factor of visualization: "+ str(result_var))'''
            
            
            