import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('adult.csv')

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
l=LabelEncoder()
o=OrdinalEncoder()

#We use ordinal encoder to Encode Independent features
for i in df.columns:
    if df[i].dtypes=='O' and i!='income':
        df[i]=o.fit_transform(df[i].values.reshape(-1,1))
        

#We use label encoder to encode label 
df['salary']=l.fit_transform(df['salary'])

from scipy.stats import zscore

def threshold():
    for i in np.arange(3,5,0.2):
        data=df.copy()
        data=data[(z<i).all(axis=1)]
        loss=(df.shape[0]-data.shape[0])/df.shape[0]*100
        print('With threshold {} data loss is {}%'.format(np.round(i,1),np.round(loss,2))) 
        
z=np.abs(zscore(df))
threshold()

"""
From above we choose threhold as 4.2 as data is precious and 
we cannot afford to lose more than 8% of data.
"""
df=df[(z<4.2).all(axis=1)]

#lets split data into catagroical and numerical feature so we can undersatnd data insight better


cols_df = pd.DataFrame(df.dtypes)
num_cols = list(cols_df[cols_df[0]=='int64'].index)
cat_cols = list(cols_df[cols_df[0]=='object'].index)
print('Numeric variables includes:','\n',num_cols)
print('\n')
print('Categorical variables includes','\n',cat_cols)


# REMOVING SKEWNESS

#using Power transformer to remove skewness
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()

for i in num_cols:
    if np.abs(df[i].skew())>0.5:
        df[i]=pt.fit_transform(df[i].values.reshape(-1,1))
        
for i in num_cols:
    sns.distplot(df[i])
    plt.figure()

#Separating dependent and independent features.
x=df.copy()
x.drop('salary',axis=1,inplace=True)
y=df['salary']

#Oversampling using Smote
from imblearn.over_sampling import SMOTE
over=SMOTE()

x,y=over.fit_resample(x,y)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
y.value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(y)
y.value_counts()
"""
Data is balanced now, both the category of income have 50% data each.
"""