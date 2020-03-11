print ("IMPORTING LIBRARIES...")
#pip install xgboost
#pip install scikit-learn
#pip install scikit-image

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb


import numpy as np

from xgboost.sklearn import XGBClassifier

#from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
#from sklearn.cross_validation import StratifiedKFold

from scipy.stats import randint, uniform

import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import requests
from requests.auth import HTTPBasicAuth


#DOWLOADING FILE FROM DROPBOX FIRST TIME
import urllib.request as urllib2
import os.path
import time
import random
while not os.path.exists('dev.csv') or not os.path.exists('oot0.csv'):
    time.sleep (3*random.random()); #Sleeping less than 3 seconds before going to Dropbox - avoid too many students at once.
    if not os.path.exists('dev.csv'):
        print ("DOWLOADING FILE dev.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://www.dropbox.com/s/yn6hvc0x9sjxbsa/dev.csv?dl=1")
        output = open('dev.csv','wb')
        output.write(csvfile.read())
        output.close()
    if not os.path.exists('oot0.csv'):
        print ("DOWLOADING FILE oot0.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://www.dropbox.com/s/i2l3iexmun0bkp2/oot0.csv?dl=1")
        output = open('oot0.csv','wb')
        output.write(csvfile.read())
        output.close()  
#DOWLOADING FILE FROM DROPBOX FIRST TIME

    
print ("LOADING DATASETS...")
df = pd.read_csv("dev.csv") #DEV-SAMPLE
dfo = pd.read_csv("oot0.csv")#OUT-OF-TIME SAMPLE

print ("IDENTIFYING TYPES...")
in_model = []
list_ib = set()  #input binary
list_icn = set() #input categorical nominal
list_ico = set() #input categorical ordinal
list_if = set()  #input numerical continuos (input float)
list_inputs = set()
output_var = 'ob_target'

for var_name in df.columns:
    if re.search('^i',var_name):
        list_inputs.add(var_name)
        print (var_name,"is input")
    if re.search('^ib_',var_name):
        list_ib.add(var_name)
        print (var_name,"is input binary")
    elif re.search('^icn_',var_name):
        list_icn.add(var_name)
        print (var_name,"is input categorical nominal")
    elif re.search('^ico_',var_name):
        list_ico.add(var_name)
        print (var_name,"is input categorical ordinal")
    elif re.search('^if_',var_name):
        list_if.add(var_name)
        print (var_name,"is input numerical continuos (input float)")
    elif re.search('^ob_',var_name):
        output_var = var_name
    else:
        print ("ERROR: unable to identify the type of:", var_name)


print ("STEP 1: DOING MY TRANSFORMATIONS...")

print ("STEP 2: SELECTING CHARACTERISTICS TO ENTER INTO THE MODEL...")
#in_model = list_inputs #['ib_var_1','icn_var_22','ico_var_25','if_var_65']
in_model = [ 'ib_var_1', 'ib_var_2', 'ib_var_3', 'ib_var_4', 'ib_var_5', 'ib_var_6',
       'ib_var_7', 'ib_var_8', 'ib_var_9', 'ib_var_10', 'ib_var_11',
       'ib_var_12', 'ib_var_13', 'ib_var_14', 'ib_var_15', 'ib_var_16',
       'ib_var_17', 'ib_var_18', 'ib_var_19', 'ib_var_20', 'ib_var_21',
       'icn_var_22', 'icn_var_23', 'icn_var_24', 'ico_var_25', 'ico_var_26',
       'ico_var_27', 'ico_var_28', 'ico_var_29', 'ico_var_30', 'ico_var_31',
       'ico_var_32', 'ico_var_33', 'ico_var_34', 'ico_var_35', 'ico_var_36',
       'ico_var_37', 'ico_var_38', 'ico_var_39', 'ico_var_40', 'ico_var_41',
       'ico_var_42', 'ico_var_43', 'ico_var_44', 'ico_var_45', 'ico_var_46',
       'ico_var_47', 'ico_var_48', 'ico_var_49', 'ico_var_50', 'ico_var_51',
       'ico_var_52', 'ico_var_53', 'ico_var_54', 'ico_var_55', 'ico_var_56',
       'ico_var_57', 'ico_var_58', 'ico_var_59', 'ico_var_60', 'ico_var_61',
       'ico_var_62', 'ico_var_63', 'ico_var_64', 'if_var_65', 'if_var_66',
       'if_var_67', 'if_var_68', 'if_var_69', 'if_var_70', 'if_var_71',
       'if_var_72', 'if_var_73', 'if_var_74', 'if_var_75', 'if_var_76',
       'if_var_77', 'if_var_78', 'if_var_79', 'if_var_80', 'if_var_81'
            ]

df.fillna(df.mean())
dfo.fillna(dfo.mean())
df.fillna(df.mean(), inplace=True)
dfo.fillna(dfo.mean(), inplace=True)

print ("STEP 3: DEVELOPING THE MODEL...")
X = df[in_model]
y = df[output_var]

Xo = dfo[in_model]

#1377
#1303
# 'ico_var_47' this feature is not very useful, as if it is removed, the score is barely affected by .03 points. 
#model

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(bootstrap=True, 
                            criterion='entropy',
                       max_features='log2',
                       oob_score=True,
                       min_samples_leaf=1, 
                       min_samples_split=2,
                       n_estimators=10000,
                       n_jobs=4, 
                       random_state=1309)
model1 = rf.fit(X,y)
y_pred = model1.predict_proba(X)[:,1]




    
print ("STEP 4: ASSESSING THE MODEL...")
# CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE
from sklearn.metrics import roc_auc_score
gini_score = 2*roc_auc_score(y, y_pred)-1
print ("GINI DEVELOPMENT=", gini_score)

def KS(b,a):  
    """Function that received two parameters; first: a binary variable representing 0=good and 1=bad, 
    and then a second variable with the prediction of the first variable, the second variable can be continuous, 
    integer or binary - continuous is better. Finally, the function returns the KS Statistics of the two lists."""
    try:
        tot_bads=1.0*sum(b)
        tot_goods=1.0*(len(b)-tot_bads)
        elements = zip(*[a,b])
        elements = sorted(elements,key= lambda x: x[0])
        elements_df = pd.DataFrame({'probability': b,'gbi': a})
        pivot_elements_df = pd.pivot_table(elements_df, values='probability', index=['gbi'], aggfunc=[sum,len]).fillna(0)
        max_ks = perc_goods = perc_bads = cum_perc_bads = cum_perc_goods = 0
        for i in range(len(pivot_elements_df)):
            perc_goods =  (pivot_elements_df.iloc[i]['len'] - pivot_elements_df.iloc[i]['sum']) / tot_goods
            perc_bads = pivot_elements_df.iloc[i]['sum']/ tot_bads
            cum_perc_goods += perc_goods
            cum_perc_bads += perc_bads
            A = cum_perc_bads-cum_perc_goods
            if abs(A['probability']) > max_ks:
                max_ks = abs(A['probability'])
    except:
        max_ks = 0
    return max_ks


KS_score = KS(y,y_pred)
print ("KS DEVELOPMENT=", KS_score) 


## Run till here to not submit


print ("STEP 5: SUBMITTING THE RESULTS...")
yo_pred  = model1.predict_proba(Xo)[:,1]
dfo['pred'] = yo_pred
dfo_tosend = dfo[list(['id','pred'])]

i=1
filename = "group_Z_sub"+str(i)+".csv"
dfo_tosend.to_csv(filename, sep=',')

url = 'http://mfalonso.pythonanywhere.com/api/v1.0/uploadpredictions'

files = {'file': (filename, open(filename, 'rb'))}
rsub = requests.post(url, files=files, auth=HTTPBasicAuth('m.s.subaiti', 'Abc123!!!'))
resp_str = str(rsub.text)
print ("RESULT SUBMISSION: ", resp_str)

import pandas as pd
df.to_excel('data_set_2.xlsx')

