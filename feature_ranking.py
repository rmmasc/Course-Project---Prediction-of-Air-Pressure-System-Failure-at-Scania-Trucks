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

# visualizing the most important features using feature imporance from Random Sampling

temp = classifier(tr_data, tr_labels,test_data, test_labels)
temp.perfRF()
arr = temp.model.feature_importances_
book = {}
features = tr_data.columns
for i in range(0,131):
    book[features[i]] = arr[i]
    
sortbook = sorted(book.items() , reverse=True, key=lambda x: x[1])
srtbook = np.array(sortbook)

yax = srtbook[:12,0]
xax = srtbook[:12,1]

print("Top ten features based on importance:")
print(srtbook)
yax = np.flip(yax)
fig, ax = plt.subplots(figsize=(8,8))

y_pos = np.arange(len(yax))
ax.barh(y_pos, xax, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(yax)
ax.set_ylabel('Feature Names')
ax.set_title('Feature Importance')

# ranking of each feature based on recursive feature elimination
'''

model = RandomForestClassifier()
rfe = RFE(model, 12)
rfe = rfe.fit(tr_data, tr_labels)
# summarize the selection of the attributes
print(rfe.support_)
print("RANKING OF EACH FEATURE BASED ON INDEX")
print(rfe.ranking_)
a = rfe.ranking_
print("Features ranked as most important")
for i in range(0,131):
    if a[i] == 1:
        print(features[i])

#Feature Importance based on wieghts in SVM with linear kernel
        
model = svm.SVC(kernel="linear",gamma="scale",probability=True)
model.fit(tr_data,tr_labels)
coef = model.coef_.copy()
coeflst = np.array([i**2 for i in coef])
coefl = coeflst.copy()
coefl = coefl.ravel()
#f = coefl.argsort()[-12:][::-1]
f = np.argpartition(coefl, -12)[-15:]
print("Top 15 features based on SVM linear kernel weights:")
for i in f:
    print(features[i])

'''