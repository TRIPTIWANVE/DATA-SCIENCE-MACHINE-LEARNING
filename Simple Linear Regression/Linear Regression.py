import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import math

original_train_set = pd.read_csv("../DATASET/LR_train.csv")
original_test_set = pd.read_csv('../DATASET/LR_test.csv')
print(original_test_set)
print(original_train_set)

train_set = original_train_set.dropna()
test_set = original_test_set.dropna()
print(train_set.shape)

X = train_set[['x']].values
Y = train_set[['y']].values

Xtest = test_set[['x']].values
Ytest = test_set[['y']].values

plt.title('Relationship between x and y of training set')
plt.scatter(X,Y,s=5,c='black')
plt.xlabel('training_set_X')
plt.ylabel('training_set_Y')
plt.show()

lm=linear_model.LinearRegression()
lm.fit(X,Y)

print('Coeff of determination:',lm.score(X,Y))
print('correlation is:',math.sqrt(lm.score(X,Y)))

p=lm.predict(X)
plt.title('Relation between predicted values and actual values in training set')
plt.scatter(Y,p,s=5)
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.show()

pr=lm.predict(Xtest)
plt.title('plot between actual values and predicted values in the test set')
plt.scatter(Ytest,pr,s=5,c='green')
plt.xlabel('test values')
plt.ylabel('predicted values')
plt.show()

