![Apache License](https://img.shields.io/hexpm/l/apa)  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)    ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)   ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![Made with matplotlib](https://user-images.githubusercontent.com/86251750/132984208-76ce70c7-816d-4f72-9c9f-90073a70310f.png)  ![seaborn](https://user-images.githubusercontent.com/86251750/132984253-32c04192-989f-4ebd-8c46-8ad1a194a492.png)  ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white)  ![coursera](https://img.shields.io/badge/Coursera-0056D2?style=for-the-badge&logo=Coursera&logoColor=white) ![udemy](https://img.shields.io/badge/Udemy-EC5252?style=for-the-badge&logo=Udemy&logoColor=white)

## SALARY PREDICTION

With more and more focus on knowledge based industry,
the compensation planning for the human capital is
becoming a key strategic area for companies to ensure
sustained growth and success. One of the problems which
corporations face today is the challenge of retaining highperforming employees and also hire talented people from
other organizations. In both the cases, salary happens to be
a key decisive factor for enticing current as well as
prospective employees. Hence an optimal salary offer which is win-win for both the employee (current as well as
prospective) and the company, is extremely important for
retaining or attracting employees to any organization.
Human resource managers have long realized that many
factors affect the salary expectation of an employee and
only her past performance or performance during interview
is not the sole determiner of her expected salary. Hence
recruiters need to weigh various factors including
demographic as well as others to make final offer to an
employee.

The census dataset from UCI (University of California,
1994) contains fifteen demographic attributes/features for
each member of a population of size 32,561 including their
individual salary class. The dataset is not specific to any
company and that is why it doesn’t contain any
performance attribute. There are two possible salary classes
for a person, namely, greater than US$ 50K (>50K) and less
than or equal to US$ 50K (<=50K). This dataset is not
balanced in terms of the numbers of these two salary classes
as approximately 30% of the tuples in the dataset belong
to >50K class and rest 70% tuples belong to <=50K
category. The dataset contains data primarily for male
workers (21,790) of private companies who belong to white
category (27,816). In terms of education levels, the dataset
represents all kinds of categories, namely, bachelors, HS,
masters, doctorate etc.
 The basic problem is to find out a classification algorithm
which will result in maximum accuracy in prediction of
salary class (>50K, <=50K) based on the given set (or
subset) of attributes. Hence the objectives are
the following:

• Run various classification engines on the UCI
census dataset

• Compare prediction performance of various
classification engines in terms of precision, recall, true positive rate (TP Rate), false positive rate (FP
Rate), F-measure and area under ROC curve

• Assess impact of feature selection techniques on
quality of results and find out if a subset of
features can be sufficient for training instead of the
full set for achieving optimal performance.


## Acknowledgements

 - [python for ML and Data science, udemy](https://www.udemy.com/course/python-for-machine-learning-data-science-masterclass)
 - [ML A-Z, udemy](https://www.udemy.com/course/machinelearning/)
 - [ML by Stanford University ](https://www.coursera.org/learn/machine-learning)



## Appendix

* [Aim](#aim)
* [Dataset used](#data)
* [Exploring the Data](#viz)
   - [Dashboard](#dashboard)
   - [Matplotlib](#matplotlib)
   - [Seaborn](#seaborn)
* [feature engineering](#fe)
* [prediction with various models](#models)
* [conclusion](#conclusion)

## AIM:<a name="aim"></a>

Predict whether income exceeds $50K/yr based on census data

## Dataset Used:<a name="data"></a>

This dataset has been taken from [uci](https://archive.ics.uci.edu/ml/datasets/Adult)

Listing of attributes:

>50K, <=50K.

• age: continuous.

• workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.

• fnlwgt: continuous.

• education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

• education-num: continuous.

• marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.

• occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.

• relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.

• race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.

• sex: Female, Male.

• capital-gain: continuous.

• capital-loss: continuous.

• hours-per-week: continuous.

• native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Exploring the Data:<a name="viz"></a>

I have used pandas, matplotlib and seaborn visualization skills.

**Matplotlib:**<a name="matplotlib"></a>
--------
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.You can draw up all sorts of charts(such as Bar Graph, Pie Chart, Box Plot, Histogram. Subplots ,Scatter Plot and many more) and visualization using matplotlib.

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install matplotlib:

    pip: pip install matplotlib

    anaconda: conda install matplotlib
    
    import matplotlib.pyplot as plt

![matplotlib](https://eli.thegreenplace.net/images/2016/animline.gif)

for more information you can refer to [matplotlib](https://matplotlib.org/) official site

**Seaborn:**<a name="seaborn"></a>
------
Seaborn is built on top of Python’s core visualization library Matplotlib. Seaborn comes with some very important features that make it easy to use. Some of these features are:

**Visualizing univariate and bivariate data.**

**Fitting and visualizing linear regression models.**

**Plotting statistical time series data.**

**Seaborn works well with NumPy and Pandas data structures**

**Built-in themes for styling Matplotlib graphics**

**The knowledge of Matplotlib is recommended to tweak Seaborn’s default plots.**

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install seaborn:

    pip: pip install seaborn

    anaconda: conda install seaborn
    
    import seaborn as sns
    
![seaborn](https://i.stack.imgur.com/uzyHd.gif)

for more information you can refer to [seaborn](https://seaborn.pydata.org/) official site.

**Dashboard:**<a name="dashboard"></a>
------

![89c3c877-3d86-468c-8dbf-550266936181](https://user-images.githubusercontent.com/86251750/145842396-b3f3eeeb-ad4b-4caa-89ae-350e0bb3c7b2.gif)

**Screenshots from notebook**

![download](https://user-images.githubusercontent.com/86251750/145843305-46437d17-cafd-48af-8c27-d90738b239b0.png)

![download](https://user-images.githubusercontent.com/86251750/145843782-7bdbe452-bcd2-4ac3-bad4-5f8269f08acd.png)

![download](https://user-images.githubusercontent.com/86251750/145843966-a29522ef-a146-4a6b-9ca6-554887b97514.png)

![download](https://user-images.githubusercontent.com/86251750/145844209-7f42ca07-ccba-4919-9861-444776456db9.png)

## My approaches on Feature Engineering<a name="fe"></a>
-------

* I use ordinal encoder to Encode Independent features.
* I use label encoder to encode label i.e., my target variable.
* used zscore with a threhold as 4.2 to remove some data as data is precious and we cannot afford to lose more than 8% of data.
* removed skewness using Power transformer.
* Separating dependent and independent features.
* class resampling or Oversampling using Smote to balance, both the category of income to have 50% data each.
* scale the data using MinMax scaler.
* Finally used various classification model for predicion and choose the best model as my final model.

## Prediction with various Models:<a name="models"></a>
------

I have used various classification models for the prediction.

**GaussianNB()**

    Accuracy 0.7834
    Mean of Cross Validation Score 0.7782
    AUC_ROC Score 0.8644

                 precision    recall  f1-score   support

           0       0.76      0.80      0.78      5627
           1       0.80      0.77      0.79      6085

    accuracy                           0.78     11712
    
**KNeighborsClassifier()**
       
    Accuracy 0.8501
    Mean of Cross Validation Score 0.8588
    AUC_ROC Score 0.9165

              precision    recall  f1-score   support

           0       0.79      0.90      0.84      5172
           1       0.91      0.81      0.86      6540

    accuracy                           0.85     11712

**LogisticRegression()**

    Accuracy 0.7769
    Mean of Cross Validation Score 0.7667
    AUC_ROC Score 0.8534

              precision    recall  f1-score   support

           0       0.76      0.79      0.77      5631
           1       0.80      0.77      0.78      6081

    accuracy                           0.78     11712

**DecisionTreeClassifier()**
    
    Accuracy 0.8526
    Mean of Cross Validation Score 0.8544
    AUC_ROC Score 0.8527

              precision    recall  f1-score   support

           0       0.85      0.85      0.85      5852
           1       0.85      0.85      0.85      5860

    accuracy                           0.85     11712


**RandomForestClassifier()**

    Accuracy 0.9008
    Mean of Cross Validation Score 0.9019
    AUC_ROC Score 0.9643

              precision    recall  f1-score   support

           0       0.89      0.91      0.90      5688
           1       0.92      0.89      0.90      6024

    accuracy                           0.90     11712

**GradientBoostingClassifier()**

    Accuracy 0.8688
    Mean of Cross Validation Score 0.8669
    AUC_ROC Score 0.9485

              precision    recall  f1-score   support

           0       0.84      0.89      0.86      5509
           1       0.90      0.85      0.87      6203

    accuracy                           0.87     11712
    
**XGBClassifier()**

    Accuracy 0.9114
    Mean of Cross Validation Score 0.9002
    AUC_ROC Score 0.9749

              precision    recall  f1-score   support

           0       0.91      0.91      0.91      5896
           1       0.91      0.91      0.91      5816

    accuracy                           0.91     11712
 

## CONCLUSION:<a name="conclusion"></a>
-----

From various model prediction we can see that Xtreme Gradient Boost give us the best performance, so I further try hyperparameter tuning on them but it looks like our xgboost model with hyperparameter tuning gives us bit lower performance with respect to default parameters of xgboost. 

So I choose XGBOOST with default paramter as my final model.

