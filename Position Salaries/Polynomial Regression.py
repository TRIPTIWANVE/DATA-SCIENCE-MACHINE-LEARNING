#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get dataset
dataset = pd.read_csv('../DATASET/Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y= dataset.iloc[:,2].values
#spliting the dataset into training set and test set
'''from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
'''
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='green')
plt.title('Truth or Bluff(LR)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff(PR)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()