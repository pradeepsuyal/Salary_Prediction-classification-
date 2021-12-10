import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('adult.csv')

#let's check that wheather there are null values
import missingno as msno
msno.matrix(df) #looks like there are not any null values

#lets see what values workclass feature contain
df['workclass'].value_counts() #looks like there are some missing value in dataset i.e., "?" 

#lets see if there are any other missing values in the feature
(df == ' ?').sum() #we can see that there are three feature  which contain missing values

#let's replace it by Nan value
df.replace(to_replace=' ?',value=np.nan,inplace=True)
df.isna().sum()

#plotting to check for Null values again
msno.bar(df, figsize=(12,8))#looks like All 3 variables with missing values are categorical in nature. 

#let's fill the missing values with the mode of the respective columns
null_columns =['workclass','occupation','country']
for i in null_columns:
    df.fillna(df[i].mode()[0], inplace=True)
    
df.isnull().sum() # all null values has been removed

#1. Correlation between numberic variables
plt.figure(figsize=(5,5))

matrix = np.triu(df.corr())   
sns.heatmap(df.corr(),cmap='rainbow',square=True,linecolor='black',linewidths=1, annot = True, mask=matrix)
plt.title('correlation between various features')
plt.savefig('correlation.png')

#lets split data into catagroical and numerical feature so we can undersatnd data insight better
cols_df = pd.DataFrame(df.dtypes)
num_cols = list(cols_df[cols_df[0]=='int64'].index)
cat_cols = list(cols_df[cols_df[0]=='object'].index)[:-1] #excluding target column of income 
print('Numeric variables includes:','\n',num_cols)
print('\n')
print('Categorical variables includes','\n',cat_cols)

#2. histogram for each feature
for i in num_cols:
    plt.figure(figsize=(6,3))
    df[df['salary']==' <=50K'][i].hist(color='gray')
    df[df['salary']==' >50K'][i].hist(color='purple')
    plt.title(i)
    plt.show()

    
import plotly.express as px
fig = px.histogram(x=df['workclass'], color=df['salary'],color_discrete_sequence=['grey','yellow'], height=400, width=700, title='Work Class VS Income',
                  labels={'Work':'Work'})
fig.show()

fig = px.histogram(x=df['occupation'], color=df['salary'],color_discrete_sequence=['grey','plum'], height=400, width=700, title='Occupation VS Income')
fig.show()

fig = px.histogram(x=df['education'], color=df['salary'], color_discrete_sequence=['grey','orange'], height=400, width=700, title='Education VS Income')
fig.show()

#Categorical Variables
for i in cat_cols:
    ct = pd.crosstab(df[i],df['salary'],margins=True, margins_name="Total")
    ct.drop(labels='Total',axis=0,inplace=True) #Removing subtotal row 
    ct.sort_values(by='Total',ascending=False,inplace=True) #Sorting based on total column
    #Selecting only top 6 categories for plotting
    ct.iloc[:6,:].plot(kind='bar',colormap='coolwarm',edgecolor='black')  
    plt.xlabel(' ')
    plt.title(str(i).capitalize())
    plt.legend(loc=1)
    plt.show()
    plt.savefig('categorical_variables.png')
    
sns.countplot(y=df['marital-status'], hue=df['salary'], order = df['marital-status'].value_counts().index)

fig = plt.figure(figsize=(15,15))
ax1= fig.add_subplot(411)
ax2= fig.add_subplot(412)
ax3= fig.add_subplot(413)
ax4= fig.add_subplot(414)

data_workclass = round(pd.crosstab(df.workclass, df.salary).div(pd.crosstab(df.workclass, df.salary).apply(sum,1),0),2)
data_occupation = round(pd.crosstab(df.occupation, df.salary).div(pd.crosstab(df.occupation, df.salary).apply(sum,1),0),2)

## Setting space between both subplots
plt.subplots_adjust(left=None,
                    bottom=None, 
                    right=None, 
                    top=1, 
                    wspace=None, 
                    hspace=0.5)

## Grapphing
sns.countplot(x='workclass', hue='salary', data= df, ax=ax1,)
data_workclass.plot.bar(ax=ax2, edgecolor='w',linewidth=1.3)

sns.countplot(x='occupation', hue='salary', data= df, ax=ax3)
data_occupation.plot.bar(ax=ax4, edgecolor='w',linewidth=1.3)

## Removing lines from the graph
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['bottom'].set_visible(False)


## Removing subplots legends
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=30, ha='right')
ax3.get_legend().remove()
ax4.get_legend().remove()



## Title
ax1.set_title("Workclass", loc='center',fontweight='bold',fontsize=14)
ax3.set_title("Occupation", loc='center',fontweight='bold',fontsize=14)
ax2.set_title("Ratio of Workclass", loc='center',fontweight='bold',fontsize=14)
ax4.set_title("Ratio of Occupation", loc='center',fontweight='bold',fontsize=14)
ax1.set_xlabel(" ")
ax1.set_ylabel(' ')
ax2.set_xlabel(" ")
ax2.set_ylabel(' ')
ax3.set_xlabel(" ")
ax3.set_ylabel(' ')
ax4.set_xlabel(" ")
ax4.set_ylabel(' ')


## Legend
line_labels = ["<=50K", ">50K"]
fig.legend(
    loc="upper right",
    labels=line_labels) 

#age vs Categorical features
"""
Individuals working in the government secctor have atmost age 70
to 80 with few outliers which must be the retirement age for them.
There are no individuals who do not work after age of 30. There
are no individuals of age >70 belonging to the pre school 
education category while Doctorates and proffessors appear from 
late 20's as they have to study for more years to get to that 
level of education. Same is the case with education num, as the 
education number increases age also is increased. There are no 
people after the age of 50 in the married to armed forces 
category with just a few outliers. Widowed category has seen 
increase as the age age seem to increase, there are very few 
widows at an early age. There are less people with high age
from other races than the white race. There are more no. of 
working men at higher age than women. There 
are very few people belonging from other countries with high age.

"""
fig,ax=plt.subplots(5,2,figsize=(15,55))
r=0
c=0
for i,n in enumerate(cat_cols):
    if i%2==0 and i>0:
        r+=1
        c=0
    graph=sns.stripplot(x=n,y='age',data=df,ax=ax[r,c], palette = 'ocean')
    if n=='native-country' or n=='occupation' or n=='education':
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 90)
    else:
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 45)
    if n!='education-num':
         graph.set(xlabel=None)
    c+=1
    
    
#Hours per week vs categorical Feature
"""
Govt employees do not work more than 80 hours a week that also 
with rare cases. It is seen that people with less education worl
more no. hours of the week which is quite logical. No armed 
force person works more than 60 hours a week while farmers and
transport movers has working hours mean higher than other 
occupation. More no, of individuals who have relationship as own
child have high density for working only 20 hous 
a week. Female works for less no. of hours as compared to men.

"""
fig,ax=plt.subplots(5,2,figsize=(15,55))
r=0
c=0
for i,n in enumerate(cat_cols):
    if i%2==0 and i>0:
        r+=1
        c=0
    graph=sns.violinplot(x=n,y='hours-per-week',data=df,ax=ax[r,c])
    if n=='native-country' or n=='occupation' or n=='education':
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 90)
    else:
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 45)
        if n!='education-num':
            graph.set(xlabel=None)
        c+=1
        
