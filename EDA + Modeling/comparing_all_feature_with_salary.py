import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('adult.csv')

"""AGE VS INCOME
The grapghs shows people belonging to diff countries have less chances of earning >50k which is wrong, 
this is because no. of individuals belonging from other countries other than U.S are very low nut it is to be noticed that there are more people in the category <=50k than >50k.
"""

plt.figure(figsize=(6,8))
sns.boxenplot(x='salary',y='age',data=df,palette="Dark2")
#People with higher mean age earn >50k while there are individuals earning <=50k even wat very high age.


"""
As the capital gain increases more people fall into >50k salary while mean of both categories remain cloase to zero capital.gain
"""
plt.figure(figsize=(6,8))
sns.boxenplot(x='salary',y='capital-gain',data=df,palette="crest")


"""
There is more density in the >50k income category with increase in capital loss while mean of both categories remain cloase to zero capital.gain
"""
plt.figure(figsize=(6,8))
sns.boxenplot(x='salary',y='capital-loss',data=df,palette="ocean")

"""
People earning >50K income work mean hours per week greater than tose earning <50K while people from both the categories work from min to max hours per week.
"""
plt.figure(figsize=(6,8))
sns.boxenplot(x='salary',y='hours-per-week',data=df,palette="rocket")

#Let us take a look at education-num and education as these variables are largely similar in nature

sns.relplot(x="education-num", y="education",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=df)


plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
df['education'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.subplot(2,1,2)
sns.countplot(x='education',data=df)
plt.xticks(rotation=45)
plt.ylabel('No. of Individuals')
df['education'].value_counts()

"""
There are 9 workclass in total including Never worked and one unknown category(?).
Most individuals work in private sector and there are very few who have never worked or work without pay. There are 3 categories of govt job provided state, federal and local among which no. of people working in the local govt is highest.
"""

