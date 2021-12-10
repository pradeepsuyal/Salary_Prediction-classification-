import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('adult.csv')


sns.barplot(x='salary',y='age',hue='sex',data=df, color = 'orange')
plt.ylabel('Average age')
"As the age increases people are paid more but males are paid more than females."

sns.factorplot(x='workclass',y='education-num',hue='salary',data=df)
plt.xticks(rotation=90)
"""
Some people belonging to a particular workclass might have less
education and some workclass might require more education level,
but no matter whatever workclass, people in the same workclass, 
if they have higher education level they earn more. It is also
to be noticed that there is no person from without pay and never 
worked workclass category who earn more than 50k which is logical.
"""

sns.factorplot(x='sex',y='education-num',hue='salary',data=df,)
plt.xticks(rotation=90)
"Females with higher education level earn equal to men having less education level than them irrespective of any income category they fall."

sns.factorplot(x='race',y='education-num',hue='salary',data=df)
plt.xticks(rotation=90)
"""
Asian pacific race have comparatively more education than the 
fellows who earn same as much as they do, belonging to other 
races. Indians 
and some other races earn >50k with lowest education level.
"""

sns.factorplot(x='occupation',y='education-num',hue='salary',data=df)
plt.xticks(rotation=90)
"""
People with highest education level belong to armed forces, 
but people with even education level quite low, who belong to
handlers cleaners, transport moving occupation earn as much as
they do. Same is the case with prof speciality. occupation of 
private house service who earn >50k and <50k have the highest 
education level difference while prof speciality have the minimum 
difference.
"""
