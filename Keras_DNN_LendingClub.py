# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:03:36 2021

@author: wikto
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Helper Functions
def feat_info(col_name):
    try:
        print(data_info.loc[col_name]['Description'])
    except KeyError:
        print(f'KeyError: {col_name} not found')

#Main
DATA_INFO_PATH = r'C:\Users\wikto\Downloads\Py_DS_ML_Bootcamp-master\Refactored_Py_DS_ML_Bootcamp-master\TensorFlow_FILES/DATA/lending_club_info.csv'
data_info = pd.read_csv(DATA_INFO_PATH, index_col='LoanStatNew')

feat_info('mort_acc')
DATA_PATH = r'C:\Users\wikto\Downloads\Py_DS_ML_Bootcamp-master\Refactored_Py_DS_ML_Bootcamp-master\TensorFlow_FILES/DATA/lending_club_loan_two.csv'
df = pd.read_csv(DATA_PATH)

sns.countplot(df['loan_status'])
sns.displot(df['loan_amnt'], bins=20)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True,cmap='GnBu')
feat_info('installment')

df['loan_repaid'] = [1 if x=='Fully Paid' else 0 for x in df['loan_status']]
df.corr()['loan_repaid'][:-1].plot.bar()
df.isna().sum()
df.isna().sum()/len(df)*100

# ~~FIG empt_length countplot
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order)

#~~FIG loan status per emp_length
charged_off_pr = df[df['loan_status']=='Charged Off']['emp_length'].value_counts() / df['emp_length'].value_counts() 
charged_off_pr.plot(kind='bar')

#~~DROP
df.drop('emp_length', inplace = True, axis=1)   #All classes have similar charged off percentage
df.drop('emp_title', inplace = True, axis=1)    #too many unique values
df.drop('title', inplace = True, axis=1)        #repetition with 'purpose' feature
