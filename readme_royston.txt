1.Author Information: 
   Name: Royston Marian Mascarenhas 
   ID: 8286328166
   Email: rmascare@usc.edu

2.Compiler and platform:
   Language: Python 3
   Compiler/IDE: Spyder/Jupyter Notebook
   OS: Windows 10
  
FEEL FREE TO UNCOMMENT AND RUN.
SOME PARTS HAVE BEEN COMMMENTED TO PREVENT RUN TIME FROM EXCEEDING BARE MINIMUM.
THEIR PURPOSE HAS BEEN STATED AS A COMMENT. FEEL FREE TO UNCOMMENT AND RUN.

There are 3 class files:
1. Transform.py
2. Preprocess.py
3. classifier.py

Please execute main.py first

For data analysis or feature ranking, execute main.py to generate relevant pickle files and then 
execute data_analysis.py or feature_ranking.py.

I have included the pickle files too, so to avoid preprocessing runtime, 
you can just change histflag to True in main.py line 43. 

Please note that 
If util file is not found, execute the following imports:

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:51:06 2019

@author: royma
"""

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

Please contact me rmascare@usc.edu if help is needed


