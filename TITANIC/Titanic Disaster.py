#1) Import Necessary Libraries
#Import Libraries
#data analysis libraries
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#2) Read in and Explore the Data
#import train and test CSV files
train = pd.read_csv("../DATASET/Titanic_train.csv")
test = pd.read_csv("../DATASET/Titanic_test.csv")

#take a look at the training data
print(train.describe(include="all"))

#3) Data Analysis
#get a list of the features within the dataset

print(train.columns)

#see a sample of the dataset to get an idea of the variables

print(train.sample(5))

#Numerical Features: Age (Continuous), Fare (Continuous), SibSp (Discrete), Parch (Discrete)
#Categorical Features: Survived, Sex, Embarked, Pclass
#Alphanumeric Features: Ticket, Cabin

#check for any other unusable values
print(pd.isnull(train).sum())

#4) Data Visualization
# here i am not visualizing the data
#5) Cleaning Data
print(test.describe(include="all"))

#we'll start off by dropping the Cabin feature since not a lot more useful information can be extracted from it.

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

#we can also drop the Ticket feature since it's unlikely to yield any useful information

train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

#now we need to fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)

#replacing the missing values in the Embarked feature with S
train = train.fillna({"Embarked": "S"})

#create a combined group of both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train['Title'], train['Sex']))

# replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train.head())

# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["Age"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["Age"].mode() #Student
mrs_age = train[train["Title"] == 3]["Age"].mode() #Adult
master_age = train[train["Title"] == 4]["Age"].mode() #Baby
royal_age = train[train["Title"] == 5]["Age"].mode() #Adult
rare_age = train[train["Title"] == 6]["Age"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}


for x in range(len(train["Age"])):
    if train["Age"][x] == "Unknown":
        train["Age"][x] = age_title_mapping[train["Title"][x]]

for x in range(len(test["Age"])):
    if test["Age"][x] == "Unknown":
        test["Age"][x] = age_title_mapping[test["Title"][x]]


#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['Age'] = train['Age'].map(age_mapping)
test['Age'] = test['Age'].map(age_mapping)

print(train.head())

#dropping the Age feature for now, might change
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
#drop the name feature since it contains no more useful information.
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

print(train.head())
#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

print(train.head())

# fill in missing Fare value in test set based on mean fare for that Pclass
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x]  # Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

# map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])

# drop Fare values
train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)
print(train.head())
print(test.head())

#6) Choosing the Best Model
#Splitting the Training Data
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.3, random_state = 0)
# Testing Different Models
# I will be testing the following models with my training data:
# Gaussian Naive Bayes
# Logistic Regression
# Support Vector Machines
# Perceptron
# Decision Tree Classifier
# Random Forest Classifier
# KNN or k-Nearest Neighbors
# Stochastic Gradient Descent
# Gradient Boosting Classifier

print("-"*100)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Naive Bayes",acc_gaussian)
print("-"*100)


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Logistic Regression",acc_logreg)
print("-"*100)

# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print("SVM",acc_svc)
print("-"*100)

# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print("LINEAR SVC",acc_linear_svc)
print("-"*100)

# Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Perceptron",acc_perceptron)
print("-"*100)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Decision Tree",acc_decisiontree)
print("-"*100)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print("RandomForest",acc_randomforest)
print("-"*100)

# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print("K-NN",acc_knn)
print("-"*100)

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print("SGD",acc_sgd)
print("-"*100)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print("GradientBoostingClassifier",acc_gbk)
print("-"*100)

#all in a dictionary
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC',
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg,
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
print(models.sort_values(by='Score', ascending=False))